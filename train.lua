require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'logroll'

require 'model_utils'
require 'TripleDataLoader'
require 'AdaGrad'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Knowledge Graph Embedding Learning')
cmd:text()
cmd:text('Options')
cmd:option('-numEnt',14951,'number of entities in the knowledge graph')
cmd:option('-numRel',1345,'number of relations in the knowledge graph')
cmd:option('-entSize',256,'size of the entity embedding')
cmd:option('-relSize',256,'size of the relation embedding')
cmd:option('-nfilter',256,'number of filters in the score model')
cmd:option('-batchSize',4096,'number of triples in a mini-batch')
cmd:option('-negSize',4,'number of corrupted instance for each triple in a mini-batch')
cmd:option('-initRange',0.1,'the uniform parameter initialization range')
cmd:option('-maxEpochs',100,'number of full passes through the training data')
cmd:option('-costMargin',1,'the margin used in the ranking cost')
cmd:option('-useGPU',1,'whether to use gpu for computation')
cmd:option('-printEvery',0,'how many steps/minibatches between printing out the loss')
cmd:option('-saveEvery',10,'how many epochs between auto save trained models')
cmd:option('-logFile','Logs/knowledge.embedding.log','log file to record training information')
cmd:option('-dataFile', 'Data/traindata.t7', 'training data file')
cmd:text()

local opt = cmd:parse(arg)
if opt.useGPU > 0 then
	torch.setdefaulttensortype('torch.CudaTensor')
end

function TripleUnit(entSize, relSize)
	local sub = nn.Identity()()
	local rel = nn.Identity()()
	local obj = nn.Identity()()
		
	local joinInput = nn.JoinTable(1, 1) ({sub, rel, obj})
	
	local gatesActs = nn.Sigmoid() (nn.Linear(2*entSize+relSize, 2*entSize+relSize) (joinInput))

	local tripleVec = nn.CMulTable() ({joinInput, gatesActs})

	return nn.gModule({sub, rel, obj}, {tripleVec})
end

-- logger
-- local logger = logroll.print_logger()
local logger = logroll.file_logger(opt.logFile)

-- Master Model
local EntityEmbed   = cudacheck(nn.LookupTable(opt.numEnt, opt.entSize))
local entParams, entGrads = EntityEmbed:getParameters()
entParams:uniform(-opt.initRange,opt.initRange)
local RelationEmbed = cudacheck(nn.LookupTable(opt.numRel, opt.relSize))
local relParams, relGrads = RelationEmbed:getParameters()
relParams:uniform(-opt.initRange,opt.initRange)

local TripleEmbed = cudacheck(TripleUnit(opt.entSize, opt.relSize))
local ScoreModel = nn.Sequential()
ScoreModel:add(TripleEmbed)
ScoreModel:add(nn.Linear(2*opt.entSize+opt.relSize, opt.nfilter))
ScoreModel:add(nn.Sigmoid())
ScoreModel:add(nn.Linear(opt.nfilter, 1))
ScoreModel = cudacheck(ScoreModel)

scoreParams, scoreGrads = ScoreModel:getParameters()

local model = {}
model.EntityEmbed   = EntityEmbed
model.RelationEmbed = RelationEmbed
model.ScoreModel    = ScoreModel

-- Clone models with parameter sharing
local SubEmbed, ObjEmbed, NegSubEmbed, NegObjEmbed = unpack(cloneManyTimes(EntityEmbed, 4))
local RelEmbed, NegRelEmbed = unpack(cloneManyTimes(RelationEmbed, 2))
local PositiveModel, NegObjModel, NegSubModel, NegRelModel = unpack(cloneManyTimes(ScoreModel, 4))

-- Citerion
local objCriterion = cudacheck(nn.MarginRankingCriterion(opt.costMargin))
local subCriterion = cudacheck(nn.MarginRankingCriterion(opt.costMargin))
local relCriterion = cudacheck(nn.MarginRankingCriterion(opt.costMargin))

-- Data Loader
local loader = TripleDataLoader(opt.dataFile, opt.batchSize, opt.negSize, logger)
local zeroTensor = torch.zeros(loader.batchSize*loader.negSize*(opt.relSize+2*opt.entSize))

-- Optimizer
-- Configurations for AdaGrad
local aoptimParams, aoptimGrads = {scoreParams, entParams, relParams}, {scoreGrads, entGrads, relGrads}
local aoptimConf = {lr = {4e-2, 4e-2, 4e-2}, logger = logger}
local aoptimizer = AdaGrad(aoptimGrads, aoptimConf)

local maxIter = math.floor(loader.numData / opt.batchSize) * opt.maxEpochs
local epochLoss, cummLoss = 0, 0
for i = 1, maxIter do
	xlua.progress(i, maxIter)
	EntityEmbed:zeroGradParameters()
	RelationEmbed:zeroGradParameters()
	scoreGrads:zero()

	local subIdx, relIdx, objIdx, negSubIdx, negObjIdx, negRelIdx = loader:nextBatch()

	-- forward pass
	local sub = SubEmbed:forward(subIdx)
	local obj = ObjEmbed:forward(objIdx)
	local rel = RelEmbed:forward(relIdx)

	local negObj = NegObjEmbed:forward(negObjIdx)
	local negSub = NegSubEmbed:forward(negSubIdx)
	local negRel = NegRelEmbed:forward(negRelIdx)

	local posScore    = PositiveModel:forward({sub, rel, obj})
	local negObjScore = NegObjModel:forward({sub, rel, negObj})
	local negSubScore = NegSubModel:forward({negSub, rel, obj})
	local negRelScore = NegRelModel:forward({sub, negRel, obj})

	-- criterion
	local lossObj = objCriterion:forward({posScore, negObjScore}, 1):sum()
	local lossSub = subCriterion:forward({posScore, negSubScore}, 1):sum()
	local lossRel = relCriterion:forward({posScore, negRelScore}, 1):sum()

	epochLoss = epochLoss + lossObj + lossSub + lossRel
	cummLoss  = cummLoss  + lossObj + lossSub + lossRel

	local d_objScoreTable = objCriterion:backward({posScore, negObjScore}, 1)
	local d_subScoreTable = subCriterion:backward({posScore, negSubScore}, 1)
	local d_relScoreTable = relCriterion:backward({posScore, negRelScore}, 1)

	-- backward pass
	local d_posScore = d_objScoreTable[1] + d_subScoreTable[1] + d_relScoreTable[1]

	local d_posEmbedTab    = PositiveModel:backward({sub, rel, obj}, d_posScore)
	local d_negObjEmbedTab = NegObjModel:backward({sub, rel, negObj}, d_objScoreTable[2])
	local d_negSubEmbedTab = NegSubModel:backward({negSub, rel, obj}, d_subScoreTable[2])
	local d_negRelEmbedTab = NegRelModel:backward({sub, negRel, obj}, d_relScoreTable[2])

	local d_sub = d_posEmbedTab[1] + d_negObjEmbedTab[1] + d_negRelEmbedTab[1]
	local d_obj = d_posEmbedTab[3] + d_negSubEmbedTab[3] + d_negRelEmbedTab[3]
	local d_rel = d_posEmbedTab[2] + d_negObjEmbedTab[2] + d_negSubEmbedTab[2]

	SubEmbed:backward(subIdx, d_sub)
	ObjEmbed:backward(objIdx, d_obj)
	NegObjEmbed:backward(negObjIdx, d_negObjEmbedTab[3])
	NegSubEmbed:backward(negSubIdx, d_negSubEmbedTab[1])

	RelEmbed:backward(relIdx, d_rel)
	NegRelEmbed:backward(negRelIdx, d_negRelEmbedTab[2])
	
	aoptimizer:updateParams(aoptimParams, aoptimGrads)

	if opt.printEvery > 0 and i % opt.printEvery == 0 then
		logger.info (string.format('[iter %6d] : %f', 
			i, cummLoss / opt.printEvery / opt.batchSize / opt.negSize))
		cummLoss = 0
	end

	if i % math.floor(loader.numData / opt.batchSize) == 0 then
		local epoch = i / math.floor(loader.numData / opt.batchSize)
		logger.info (string.format('[Epoch %3d] : %f', 
			epoch, epochLoss / math.floor(loader.numData / opt.batchSize) / opt.batchSize / opt.negSize))
		epochLoss = 0
		if i % (opt.saveEvery * math.floor(loader.numData / opt.batchSize)) == 0 then
			torch.save('model.'..epoch, model)
		end
	end
end