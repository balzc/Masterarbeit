package cov;

import org.jblas.DoubleMatrix;

public class Periodic extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams;
	public Periodic(){
		numParams = 1;

	}
	public int getNumParams(){return numParams;}

	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;

		double d = Math.abs(x-xstar);
		d = Math.sin(Math.PI*d);
		return Math.exp(-2*d*d/parameters.get(0));

	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;

		double d = Math.abs(x.get(0)-xstar.get(0));

		double sin = Math.sin(Math.PI*d);


		return Math.exp(-2*sin*sin/parameters.get(0)*parameters.get(0));
	}

	public DoubleMatrix computeSingleValue(DoubleMatrix parameters, DoubleMatrix X){
		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.rows+" instead of "+numParams);

		
		double ell = parameters.get(0,0);
		DoubleMatrix tp = X.transpose();
		DoubleMatrix tmp = sin(dist(tp.transpose()).mul(Math.PI));
		tmp = tmp.mul(tmp);
		DoubleMatrix A = exp(tmp.mul(-2/ell*ell));

		return A;
	}

	public DoubleMatrix computeDerivatives(DoubleMatrix parameters, DoubleMatrix X, int index) {

		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.columns+" instead of "+numParams);
		if(index>numParams-1)
			throw new IllegalArgumentException("Wrong hyperparameters index "+index+" it should be smaller or equal to "+(numParams-1));

		double ell = parameters.get(0,0);
		DoubleMatrix tp = X.transpose();
		DoubleMatrix tmp = sin(dist(tp.transpose()).mul(Math.PI));
		tmp = tmp.mul(tmp);
		DoubleMatrix A = exp(tmp.mul(-2/ell*ell));
		DoubleMatrix C = tmp.mul(4/ell*ell*ell);
		A = A.mmul(C);
	
		return A;
	}

	
}
