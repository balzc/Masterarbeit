package main;

import org.jblas.DoubleMatrix;

import cov.Additive;
import cov.CovarianceFunction;
import cov.Multiplicative;
import cov.Periodic;
import cov.SquaredExponential;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		GP gp = new GP(DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(),DoubleMatrix.ones(2).transpose(), new Additive(new Multiplicative(c1, c2), c2), 0.11);
		gp.test();
	}

}
