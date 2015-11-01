require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'model_utils'
require 'logroll'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Knowledge Graph Embedding Learning')
cmd:text()
cmd:text('Options')
cmd:option('-numEnt',75043,'number of entities in the knowledge graph')
cmd:option('-numRel',13,'number of relations in the knowledge graph')
cmd:option('-entSize',128,'size of the entity embedding')
cmd:option('-relSize',128,'size of the relation embedding')
cmd:option('-nfilter',128,'number of filters in the score model')
cmd:option('-maxEpochs',100,'number of full passes through the training data')
cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
cmd:option('-useGPU',1,'whether to use gpu for computation')
cmd:option('-printEvery',100,'how many steps/minibatches between printing out the loss')
cmd:option('-saveEvery',10,'how many epochs between auto save trained models')
cmd:option('-logFile','Logs/knowledge.embedding.log','log file to record training information')
cmd:option('-dataFile', 'Data/train.torch','training data file')
cmd:text()

local opt = cmd:parse(arg)

function TripleUnit(entSize, relSize)
	local sub = nn.Identity()()
	local rel = nn.Identity()()
	local obj = nn.Identity()()
	
	function readerFunc(outSize)
		local joinInput = nn.JoinTable(1, 1) ({sub, rel, obj})
		local readerAct = nn.Sigmoid() (nn.Linear(2*entSize+relSize, outSize) (joinInput))
		return readerAct
	end

	local subReader = readerFunc(entSize)
	local relReader = readerFunc(relSize)
	local objReader = readerFunc(entSize)

	local tripleVec = nn.JoinTable(1, 1) ({
		nn.CMulTable() ({subReader, sub}),
		nn.CMulTable() ({relReader, rel}),
		nn.CMulTable() ({objReader, obj})
	})

	return nn.gModule({sub, rel, obj}, {tripleVec})
end

-- Master Model
local EntityEmbed   = nn.LookupTable(opt.numEnt, opt.entSize)
local RelationEmbed = nn.LookupTable(opt.numRel, opt.relSize)

local TripleEmbed = TripleUnit(opt.entSize, opt.relSize)
local model = nn.Sequential()
model:add(TripleEmbed)
model:add(nn.Linear(2*opt.entSize+opt.relSize, opt.nfilter))
model:add(nn.Max(2))

-- Clone models with parameter sharing
local PositiveModel, NegativeModel = unpack(cloneManyTimes(model, 2))

-- Citerion
local criterion = nn.MarginRankingCriterion(opt.costMargin)

-- Fake data
local batchSize = 4

local subIndices = torch.Tensor(batchSize):random(1, opt.numEnt)
local relIndices = torch.Tensor(batchSize):random(1, opt.numRel)
local objIndices = torch.Tensor(batchSize):random(1, opt.numEnt)

local negObjIndices = torch.Tensor(batchSize):random(1, opt.numEnt)

-- Fake forward pass
local sub = EntityEmbed:forward(subIndices):clone()
local obj = EntityEmbed:forward(objIndices):clone()
local rel = RelationEmbed:forward(relIndices)

local negObj = EntityEmbed:forward(negObjIndices):clone()

local posScore = PositiveModel:forward({sub, rel, obj})
local negScore = NegativeModel:forward({sub, rel, negObj})
local loss = criterion:forward({posScore, negScore}, 1)
print (loss)
