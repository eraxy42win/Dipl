
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix
import pandas as pd
import ExcelWriter as ExcelW
import time
import numpy as np

from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Ackley,Rastrigin,Rosenbrock,Griewank,Sphere,Whitley,Zakharov,Perm,Powell,Pinter
from NiaPy.algorithms.statistics import BasicStatistics

ArrayOfNP = [10,20]#,30,50,75,100
ArrayOfBenchmarks = [Ackley(),Rastrigin()] #,Rosenbrock(),Griewank(), Sphere(), Whitley(), Zakharov(), Perm(), Powell(), Pinter()
ArrayOfnFES = [10000]#,20000,30000
ArrayOfD = [10,20,30]
# we will run Firefly Algorithm for 25 independent runs
dataframe_collection = [] 
NUM_RUNS = 1

startWholeAlgoTime = time.time()
with pd.ExcelWriter('results.xlsx') as writer:
    for Np in ArrayOfNP:
        for nFES in ArrayOfnFES:
            for D in ArrayOfD:
                tempDataFrame = []
                for BenchFunction in ArrayOfBenchmarks:
                    resultArray = []
                    rawData = np.zeros(NUM_RUNS)
                    start = time.time()
                    for i in range(NUM_RUNS):
                        task = StoppingTask(D=D, nFES=nFES, optType=OptimizationType.MINIMIZATION, benchmark=BenchFunction)
                        algo = FireflyAlgorithm(NP=Np, alpha=0.5, betamin=0.2, gamma=1.0)
                        best = algo.run(task=task)
                        rawData[i] = best[1]
                    end = time.time()
                    elapsedTime = end - start
                    processedData = BasicStatistics(rawData)
                    #statistics
                    resultArray.append(processedData.min_value())
                    resultArray.append(processedData.max_value())
                    resultArray.append(processedData.mean())
                    resultArray.append(processedData.median())
                    resultArray.append(processedData.standard_deviation())
                    resultArray.append(elapsedTime)
                    ttt = pd.DataFrame(data=[resultArray], index=[BenchFunction.Name], columns=['Min', 'Max','Mean','Median','Standard_D','Time'])
                    tempDataFrame.append(ttt)
                result = pd.concat(tempDataFrame)
                name = 'NP_' + str(Np) + ' - nFES_' + str(nFES) + ' - D_' + str(D)
                print("")
                print(name)
                print(result)
                result.to_excel(writer,sheet_name=name)

endWholeAlgoTime = time.time()
elapsedTimeWholeAlgoTime = endWholeAlgoTime - startWholeAlgoTime
print(elapsedTimeWholeAlgoTime)


            
            



