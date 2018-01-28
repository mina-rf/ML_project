# from mnist import MNIST
#
#
# mndata = MNIST('data')
# tr_imgs , tr_lbs = mndata.load_training()
# test_imgs , test_lbs = mndata.load_testing()
#
# print()

class Classifier :

    def __init__(self , f_extractor , dim_reducer , classifier):
        self.f_extractor = f_extractor
        self.dim_reducer = dim_reducer
        self.classifier = classifier

    def extract_features(self, data):
        new_data = []
        for d in data:
            d2 = self.f_extractor(d)
            new_data.append(d2)
        return new_data

    def reduce_dimension(self,data , labels):
        self.dim_reducer.fit(data , y=labels)
        return

    def train(self,train_data):
        pass




