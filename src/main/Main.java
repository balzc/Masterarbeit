package main;

import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;

import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		int noData = 10;
		double[] dataX = new double[noData];
		double[] dataY =  new double[50];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i;
		}
		for(int i=0; i< dataY.length; i++){
			dataY[i] = Math.random();
		}
		double[] dataP = {0.1,0.7,0.5};
		double[] dataTest = {11,12,13,14,15,16,17,18,19,20};
		double nl = 10;
		double noiselevel = nl;
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn = new DoubleMatrix(dataTest);
		GP gp = new GP(DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(), new SquaredExponential(), 0.11);

//		DoubleMatrix co = gp.computeCovMatrix(X, X, P);
//		DoubleMatrix samples = gp.generateSamples(X, P, nl, gp.covf);
//		printMatrix(samples);
//		printMatrix(X);
//
//		int num = 10;
//		TestMDP testmdp = new TestMDP(DoubleMatrix.zeros(co.rows) , co.transpose(), 1, num,0,10);
//		co.print();
//		testmdp.computePrices();
//		testmdp.computeProbabilityTables();
//		testmdp.computeRewards();
//		System.out.println(testmdp.priceProb.length);
//		for(int i = 0; i < testmdp.priceProb.length; i++){
//			System.out.print(i + ": ");
//
//			for(int o = 0; o < testmdp.priceProb[i].length; o++){
//				for(int j = 0; j < testmdp.priceProb[i][o].length; j++){
//					System.out.print(testmdp.priceProb[i][o][j] + " ");
//				}
//			}
//			System.out.println();
//		}
//		testmdp.solveMDP(num);
		gp.test();
	}
	
	
	public static void printMatrix(DoubleMatrix m){
		System.out.print("[");
		for(int i = 0; i< m.rows; i++){
			for(int j = 0; j< m.columns; j++){
				System.out.print(m.get(i,j));
			}
			System.out.print("; ");

		}
		System.out.println("]");
	}
}
