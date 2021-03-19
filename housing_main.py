from housing_one import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

def runModel(question_num):
    if question_num == 1: #housing
        df = pd.read_csv('Housing.csv')

        NEregression = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = False, tolerance = False, gd = False)
        NEregression.run_model()
        print('Predicted y:'+'\n', NEregression.predict(NEregression.X_test))
        print('\n'+'------------------Results when learningrate = 0.0004, tolerance = 0.005---------------------')
        GDregression1 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.0004, tolerance = 0.005)
        GDregression1.run_model('blue')
        plt.show()
        print('Predicted y:'+'\n', GDregression1.predict(GDregression1.X_test))

        print('\n'+'------------------Results when learningrate = 0.1, tolerance = 0.005---------------------')
        GDregression2 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.1, tolerance = 0.005)
        GDregression2.run_model('purple')
        print('Predicted y:'+'\n', GDregression2.predict(GDregression2.X_test))

        print('\n'+'------------------Results when learningrate = 0.01, tolerance = 0.005---------------------')
        GDregression3 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.01, tolerance = 0.005)
        GDregression3.run_model('red')
        print('Predicted y:'+'\n', GDregression3.predict(GDregression3.X_test))

        print('\n'+'------------------Results when learningrate = 0.05, tolerance = 0.005---------------------')
        GDregression4 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.05, tolerance = 0.005)
        GDregression4.run_model('green')
        print('Predicted y:'+'\n', GDregression4.predict(GDregression4.X_test))
        plt.show()

    elif question_num == 2: #yacht
        df = pd.read_csv('yachtData.csv')

        NEregression = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = False, tolerance = False, gd = False)
        NEregression.run_model()
        print('Predicted y:'+'\n', NEregression.predict(NEregression.X_test))
        print('\n'+'------------------Results when learningrate = 0.001, tolerance = 0.001---------------------')
        GDregression1 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.001, tolerance = 0.001)
        GDregression1.run_model('blue')
        plt.show()
        print('Predicted y:'+'\n', GDregression1.predict(GDregression1.X_test))

        print('\n'+'------------------Results when learningrate = 0.1, tolerance = 0.001---------------------')
        GDregression2 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.1, tolerance = 0.001)
        GDregression2.run_model('purple')
        print('Predicted y:'+'\n', GDregression2.predict(GDregression2.X_test))

        print('\n'+'------------------Results when learningrate = 0.01, tolerance = 0.001---------------------')
        GDregression3 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.01, tolerance = 0.001)
        GDregression3.run_model('red')
        # print('Predicted y:'+'\n', GDregression3.predict(GDregression3.X_test))

        print('\n'+'------------------Results when learningrate = 0.05, tolerance = 0.001---------------------')
        GDregression4 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.05, tolerance = 0.001)
        GDregression4.run_model('green')
        print('Predicted y:'+'\n', GDregression4.predict(GDregression4.X_test))
        plt.show()


    else: #3 concrete
        df = pd.read_csv('concreteData.csv')

        print('\n'+'------------------Results when learningrate = 0.0007, tolerance = 0.0001---------------------')
        GDregression1 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.0007, tolerance = 0.0001)
        GDregression1.run_model('blue')
        plt.show()
        print('Predicted y:'+'\n', GDregression1.predict(GDregression1.X_test))

        print('\n'+'------------------Results when learningrate = 0.1, tolerance = 0.0001---------------------')
        GDregression2 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.1, tolerance = 0.0001)
        GDregression2.run_model('purple')
        print('Predicted y:'+'\n', GDregression2.predict(GDregression2.X_test))

        print('\n'+'------------------Results when learningrate = 0.01, tolerance = 0.0001---------------------')
        GDregression3 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.01, tolerance = 0.0001)
        GDregression3.run_model('red')
        print('Predicted y:'+'\n', GDregression3.predict(GDregression3.X_test))

        print('\n'+'------------------Results when learningrate = 0.05, tolerance = 0.0001---------------------')
        GDregression4 = LinearRegression(df.values[:, 0:-1], df.values[:, -1], learningrate = 0.05, tolerance = 0.0001)
        GDregression4.run_model('green')
        print('Predicted y:'+'\n', GDregression4.predict(GDregression4.X_test))
        plt.show()

#Run model via corresponding question number 1-3
result = runModel(2)
print(result)
