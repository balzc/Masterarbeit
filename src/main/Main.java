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
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i;
		}
		for(int i=0; i< dataY.length; i++){
			dataY[i] = i;
		}
		double[] dataP = {1,2};
		double[] dataTest = {11,12,13,14,15,16};
		double nl = 0.5;		
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn = new DoubleMatrix(dataTest);
		GP gp = new GP(X, Y, testIn, P, c1, nl);
		DoubleMatrix samples = gp.generateSamples(X, P, nl, c1);
		gp.setup();

		gp.getPredMean().print();
		gp.getTestCov().print();
		int num = 10;
		HomeHeatingMDP testmdp = new HomeHeatingMDP(gp.getPredMean(),gp.getTestCov(),1,5);
		testmdp.work();
		testmdp.printOptPolicy();
		testmdp.printQvals();
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
