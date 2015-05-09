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
	public double computeCovariance(double x,  double[] parameters){
		
		return x;
	}
	public DoubleMatrix computeCovarianceMatrix(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		return x;
	}
	public double noise(int index1, int index2, double sn){
		return index1==index2 ? sn*sn : 0;
	}
	public DoubleMatrix computeDerivatives(DoubleMatrix loghyper, DoubleMatrix X, int index) {
		return loghyper;
	}
}
