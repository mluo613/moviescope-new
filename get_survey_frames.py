from readSplitDataSet import gather_raw_data
import random
import pickle 

testList = gather_raw_data(mode='data/raw_data_test')

random.seed(0)
selected = random.sample(testList[2], 100)
for i in selected:
    print(i)
# pickle_out = open("Test_id_list_survey.p","wb")
# pickle.dump(selected, pickle_out)
# pickle_out.close()

#print ('id', testList[2][0])
#print ('label', testList[0][0])
#print ('plot', testList[1][0])

