package main;

import org.jblas.DoubleMatrix;

import cov.CovarianceFunction;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		GP gp = new GP(DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(), new CovarianceFunction(), 0.11);
		double test = DoubleMatrix.ones(2).distance2(DoubleMatrix.zeros(2));
	}

}
