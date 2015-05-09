package cov;

import org.jblas.DoubleMatrix;

public class CovarianceFunction {
	public DoubleMatrix parameters;
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		return x;
	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		return x.distance2(xstar);
	}
	public DoubleMatrix computeDerivatives(DoubleMatrix loghyper, DoubleMatrix X, int index) {
		return loghyper;
	}
}
