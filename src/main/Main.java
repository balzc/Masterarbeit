package main;

import org.jblas.DoubleMatrix;

import cov.*;

import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		GP gp = new GP(DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(), new Matern(), 0.11);
		gp.test();
	}

}
