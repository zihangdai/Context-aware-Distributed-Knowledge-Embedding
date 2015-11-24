require 'TripleDataLoader.lua'
require '../Attention/init.lua'
require 'cutorch'

createTripleData('Data/testdata.txt',  'Data/testdata.t7')
createTripleData('Data/traindata.txt', 'Data/traindata.t7')
createTripleData('Data/validdata.txt',  'Data/validdata.t7')