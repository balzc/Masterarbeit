package test;
import gp.GP;
import main.Main;
import mdp.EVMDP;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import util.FileHandler;
import cov.Additive;
import cov.CovarianceFunction;
import cov.Matern;
import cov.Multiplicative;
import cov.Periodic;
import cov.SquaredExponential;
import static org.junit.Assert.*;
public class Testing {
	/* train GP
	 * predict for following days
	 * set up mdp
	 * find best policy
	 * heat according to policy
	 * calculate utility
	 * repeat with gathered data
	 */
	public static void doTest(){
		
	}
	
	@Test
    public void testEVMDP() {
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices2.csv";

		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData = 96;
		double stepsize = 1./96.;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length/10; i++){
			dataTest[i] = i*stepsize + noData*stepsize;
		}
		double nl = 0.05;		
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
		DoubleMatrix predictedPrices = new DoubleMatrix();
		DoubleMatrix P = new DoubleMatrix(dataP);
		CovarianceFunction cf = a1;
		int runs = 1;
		double cumulativeU = 0;
		double currentLoad = 0;
		int steps = 96;
		int trainSetSize = 1;
		int initialOffset = 0;
		double[] loads = new double[runs*steps];
		int[] actions = new int[runs*steps];

		for(int i = 0; i < runs; i++){
			double[] xTrain = new double[steps*trainSetSize];
			for(int o = 0; o < steps*trainSetSize; o++){
				xTrain[o] = o*stepsize+initialOffset*stepsize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[steps];
			for(int o = 0; o < steps; o++){
				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);

			
		
			DoubleMatrix yTrainMPrices = Main.subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
//			printMatrix(xTrainM);
//			printMatrix(xTestM);
			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
			priceGP.setup(yTrainMPrices);
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			EVMDP testmdp = new EVMDP(predMeanPrices,predVarPrices, .5,steps);

			testmdp.work();
			
			// heat according to policy and update cumulative utility
			int tmp = initialOffset+steps*trainSetSize;
			for(int o = tmp; o < tmp + steps; o++){
//				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " +  currentLoad );
				int action = testmdp.getOptPolicy()[o-tmp][testmdp.priceToState(predMeanPrices.get(o-tmp))][testmdp.loadToState(currentLoad)];
				cumulativeU += testmdp.rewards(currentLoad, action, priceSamples.get(o)+20,o);
				currentLoad = testmdp.updateLoad(currentLoad, action);

				loads[o-steps*trainSetSize-initialOffset] = currentLoad;
				actions[o-steps*trainSetSize-initialOffset] = action;
			}
			initialOffset += steps;
			if(i > 0){
				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
			} else {
				predictedPrices = predMeanPrices;
			}

		}
		Main.printMatrix(Main.subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples).add(20));
		System.out.println();

		Main.printMatrix(predictedPrices);
		System.out.println();

		System.out.print("[");
		for(int i = 0; i< loads.length; i++){
			System.out.print(loads[i] + "; ");

		}
		System.out.println("]");
		System.out.print("[");
		for(int i = 0; i< actions.length; i++){
			System.out.print(actions[i] + "; ");

		}
		System.out.println("]");
		System.out.println(cumulativeU);



    }
}