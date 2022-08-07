from stock_forecast import stock_forecast

from tpot import TPOTClassifier

class automl_forecast(stock_forecast):
    def __init__(self, stock_code = "sh000001", useage_days = 30, split_date = '2018-01-01'):
        super().__init__(stock_code, useage_days, split_date)
        super().data_smooth()
        super().data_normalization()
    
    def automl_optimize(self):
        pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                            random_state=42, verbosity=2)
        self.generate_ann_input_data()
        pipeline_optimizer.fit(self.X_train, self.Y_train[:, 1])
        print(print(pipeline_optimizer.score(self.X_test, self.Y_test[:, 1])))
        pipeline_optimizer.export('isf_pipeline.py')

if __name__ == "__main__":

    def main():
        test = automl_forecast("sh000001",50,'2018-01-01')
        test.automl_optimize()
    
    main()
        