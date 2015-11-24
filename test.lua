require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'logroll'

require 'TripleDataLoader'
require '../Attention/init.lua'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Knowledge Graph Embedding Learning')
cmd:text()
cmd:text('Options')
cmd:option('-modelFile', 'model.10', 'trained model file')
cmd:option('-dataFile', 'Data/testdata.t7', 'test data file')
cmd:option('-printEvery',10,'how many steps between printing out the information')
cmd:option('-useGPU',1,'whether to use gpu for computation')
cmd:text()

local opt = cmd:parse(arg)
if opt.useGPU > 0 then
	require 'cutorch'
	torch.setdefaulttensortype('torch.CudaTensor')
end

-- load model
local model = torch.load(opt.modelFile)
local EntityEmbed   = cudacheck(model.EntityEmbed:double())
local RelationEmbed = cudacheck(model.RelationEmbed:double())
local ScoreModel    = cudacheck(model.ScoreModel:double())

local numEnt  = EntityEmbed.weight:size(1)
local entSize = EntityEmbed.weight:size(2)
local numRel  = RelationEmbed.weight:size(1)
local relSize = RelationEmbed.weight:size(2)

-- load data
local data = torch.load(opt.dataFile)
local subIdx = data.subs
local objIdx = data.objs
local relIdx = data.rels
local dict   = {}
for k, v in pairs(torch.load('Data/traindata.t7').dict) do
	dict[k] = v
end
for k, v in pairs(torch.load('Data/testdata.t7').dict) do
	dict[k] = v
end
for k, v in pairs(torch.load('Data/validdata.t7').dict) do
	dict[k] = v
end

local rankSumFil, rankSumRaw = 0, 0
local hitFil, hitRaw = 0, 0
local n_o  , o_n  , n_n  , o_o   = 0, 0, 0, 0
local n_o_h, o_n_h, n_n_h, o_o_h = 0, 0, 0, 0
local n_o_t, o_n_t, n_n_t, o_o_t = 0, 0, 0, 0

local hitRel, rankSumRel = 0, 0

local endIdx = subIdx:size(1)

for i = 1, endIdx do
	xlua.progress(i, endIdx)
	local sub = EntityEmbed:forward(subIdx:narrow(1,i,1)):clone()
	local rel = RelationEmbed:forward(relIdx:narrow(1,i,1)):clone()
	local obj = EntityEmbed:forward(objIdx:narrow(1,i,1)):clone()

	local sortScore
	local n_o_flag, o_n_flag, n_n_flag, o_o_flag = true, true, true, true

	local objScore = ScoreModel:forward({
			sub:expand(numEnt, entSize),
			rel:expand(numEnt, relSize),
			EntityEmbed.weight
		})
	sortScore, sortIdx = objScore:view(-1):sort(1, true)
	local objRankFil, objRankRaw = 0, 0
	for j = 1, sortIdx:size(1) do
		local key = table.concat({subIdx[i],relIdx[i],sortIdx[j]}, '_')
		if dict[key] == nil or objIdx[i] == sortIdx[j] then
			objRankFil = objRankFil + 1
			if objIdx[i] == sortIdx[j] then				
				objRankRaw = j
				break
			end
		else
			n_o_flag, o_o_flag = false, false
		end
	end

	local subScore = ScoreModel:forward({
			EntityEmbed.weight,
			rel:expand(numEnt, relSize),
			obj:expand(numEnt, entSize)
		})
	sortScore, sortIdx = subScore:view(-1):sort(1, true)
	local subRankFil, subRankRaw = 0, 0
	for j = 1, sortIdx:size(1) do
		local key = table.concat({sortIdx[j],relIdx[i],objIdx[i]}, '_')
		if dict[key] == nil or subIdx[i] == sortIdx[j] then
			subRankFil = subRankFil + 1
			if subIdx[i] == sortIdx[j] then
				subRankRaw = j
				break
			end
		else
			o_n_flag, o_o_flag = false, false
		end
	end

	local relScore = ScoreModel:forward({
			sub:expand(numRel, entSize),
			RelationEmbed.weight,
			obj:expand(numRel, entSize)
		})
	sortScore, sortIdx = relScore:view(-1):sort(1, true)
	local relRankFil, relRankRaw = 0, 0
	for j = 1, sortIdx:size(1) do
		local key = table.concat({subIdx[i],sortIdx[j],objIdx[i]}, '_')
		if dict[key] == nil or relIdx[i] == sortIdx[j] then
			relRankFil = relRankFil + 1
			if relIdx[i] == sortIdx[j] then
				relRankRaw = j
				break
			end		
		end
	end

	-- print (subRankFil,relRankFil,objRankFil)
	rankSumFil = rankSumFil + subRankFil + objRankFil
	rankSumRaw = rankSumRaw + subRankRaw + objRankRaw
	hitFil = hitFil + (subRankFil < 10 and 1 or 0) + (objRankFil < 10 and 1 or 0)
	hitRaw = hitRaw + (subRankRaw < 10 and 1 or 0) + (objRankRaw < 10 and 1 or 0)
	
	hitRel = hitRel + (relRankFil < 2 and 1 or 0)
	rankSumRel = rankSumRel + relRankFil

	if o_o_flag then
		o_o_h = o_o_h + (subRankFil < 10 and 1 or 0)
		o_o_t = o_o_t + (objRankFil < 10 and 1 or 0)
		o_o = o_o + 1
	else
		if n_o_flag then
			n_o_h = n_o_h + (subRankFil < 10 and 1 or 0)
			n_o_t = n_o_t + (objRankFil < 10 and 1 or 0)
			n_o = n_o + 1
		elseif o_n_flag then
			o_n_h = o_n_h + (subRankFil < 10 and 1 or 0)
			o_n_t = o_n_t + (objRankFil < 10 and 1 or 0)
			o_n = o_n + 1
		else
			n_n_h = n_n_h + (subRankFil < 10 and 1 or 0)
			n_n_t = n_n_t + (objRankFil < 10 and 1 or 0)
			n_n = n_n + 1
		end	
	end

	if i % opt.printEvery == 0 then
		print (string.format('[%5d] %10f, %6f', i, hitRel / i, rankSumRel / i))
		print (string.format('[%5d] %10f, %6f', i, hitFil / i / 2, rankSumFil / i / 2))
		print (string.format('[%5d] %10f, %6f', i, hitRaw / i / 2, rankSumRaw / i / 2))		
	end
end
print (string.format('[o_o_h] %10f', o_o_h / o_o))
print (string.format('[o_o_t] %10f', o_o_t / o_o))
print (string.format('[n_o_h] %10f', n_o_h / n_o))
print (string.format('[n_o_t] %10f', n_o_t / n_o))
print (string.format('[o_n_h] %10f', o_n_h / o_n))
print (string.format('[o_n_t] %10f', o_n_t / o_n))
print (string.format('[n_n_h] %10f', n_n_h / n_n))
print (string.format('[n_n_t] %10f', n_n_t / n_n))

