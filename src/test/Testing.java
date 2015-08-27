package test;
import gp.GP;
import main.Main;
import mdp.EVMDP;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import simulation.Simulation;
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
	@Test
	public void doTest(){
		Simulation s = new Simulation();
		//s.work(fhprices, fhout, vminvarInput, qminInput, qmaxInput, tstartInput, tcritInput, mqInput, kwhPerUnitInput, bisd)

		s.work("/users/balz/documents/workspace/data/interpolatedPrices.csv","/users/balz/documents/workspace/data/","/users/balz/documents/workspace/data/comp.csv", 20.,33,60.,28.,1.,20.,6,10000);
//		s.work("/users/balz/documents/workspace/masterarbeit/data/interpolatedPrices.csv","/users/balz/documents/workspace/masterarbeit/data/","/users/balz/documents/workspace/masterarbeit/data/comp.csv", 27.,33,6.,28.,1.,13.,0.75,0.5);

	}
	
//	@Test
//    public void testEVMDP() {
//		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
//
//		SquaredExponential c1 = new SquaredExponential();
//		Periodic c2 = new Periodic();
//		Matern m = new Matern();
//		Multiplicative mult = new Multiplicative(c1, c2);
//		Additive a1 = new Additive(mult, m);
//		
//		int noData = 96;
//		double stepsize = 1./96.;
//		double[] dataX = new double[noData];
//		double[] dataY =  new double[noData];
//		for(int i=0; i< dataX.length; i++){
//			dataX[i] = i*stepsize;
//		}
//		for(int i=0; i< noData; i++){
//			dataY[i] = 10;
//		}
//	
//		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
//		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
//		for(int i = 0; i < dataTest.length/10; i++){
//			dataTest[i] = i*stepsize + noData*stepsize;
//		}
//		double nl = 0.05;		
//		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
//		DoubleMatrix predictedPrices = new DoubleMatrix();
//		DoubleMatrix P = new DoubleMatrix(dataP);
//		CovarianceFunction cf = a1;
//		int runs = 1;
//		double cumulativeU = 0;
//		double currentLoad = 0;
//		int steps = 12;
//		int trainSetSize = 1;
//		int initialOffset = 0;
//		double[] loads = new double[runs*steps];
//		int[] actions = new int[runs*steps];
//		DoubleMatrix priceSimple1 = DoubleMatrix.ones(3).mul(20);
//		DoubleMatrix priceSimple2 = DoubleMatrix.ones(3).mul(30);
//		DoubleMatrix priceSimple3 = DoubleMatrix.ones(3).mul(10);
//		DoubleMatrix priceSimple4 = DoubleMatrix.ones(3).mul(10);
//		DoubleMatrix priceSimple = DoubleMatrix.concatVertically(priceSimple1, priceSimple2);
//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple3);
//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple4);
//		for(int i = 0; i < runs; i++){
//			double[] xTrain = new double[steps*trainSetSize];
//			for(int o = 0; o < steps*trainSetSize; o++){
//				xTrain[o] = o*stepsize+initialOffset*stepsize;
//			}
//			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
//			double[] xTest = new double[steps];
//			for(int o = 0; o < steps; o++){
//				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
//			}
//			DoubleMatrix xTestM = new DoubleMatrix(xTest);
//
//			
//		
//			DoubleMatrix yTrainMPrices = Main.subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
////			Main.printMatrix(xTrainM);
////			Main.printMatrix(xTestM);
////			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
////			priceGP.setup(yTrainMPrices);
//			DoubleMatrix predMeanPrices = priceSimple;//priceGP.getPredMean().add(20);
//			DoubleMatrix predVarPrices = priceSimple;//priceGP.getPredVar();
//			EVMDP testmdp = new EVMDP(predMeanPrices,predVarPrices,10,steps);
//
//			testmdp.work();
//			
//			// heat according to policy and update cumulative utility
//			int tmp = initialOffset+steps*trainSetSize;
//			for(int o = 0; o < steps; o++){
//				System.out.println("O is " +o  + " " + testmdp.priceToState(predMeanPrices.get(o))+ " " +  currentLoad );
//				int action = testmdp.getOptPolicy()[o][testmdp.priceToState(predMeanPrices.get(o))][testmdp.loadToState(currentLoad)][0];
//				cumulativeU += testmdp.rewards(currentLoad, action, priceSimple.get(o),o,0);
//				currentLoad = testmdp.updateLoad(currentLoad, action);
//
//				loads[o] = currentLoad;
//				actions[o] = action;
//			}
//			initialOffset += steps;
//			if(i > 0){
//				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
//			} else {
//				predictedPrices = predMeanPrices;
//			}
//
//		}
//		Main.printMatrix(Main.subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples).add(20));
//		System.out.println();
//
//		Main.printMatrix(predictedPrices);
//		System.out.println();
//		System.out.print("[");
//		for(int i = 0; i< loads.length; i++){
//			System.out.print(loads[i] + "; ");
//
//		}
//		System.out.println("]");
//		System.out.print("[");
//		for(int i = 0; i< actions.length; i++){
//			System.out.print(actions[i] + "; ");
//
//		}
//		System.out.println("]");
//		System.out.println(cumulativeU);
//
//
//
//    }
	
	
}
