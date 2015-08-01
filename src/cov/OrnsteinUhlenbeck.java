package cov;

import org.jblas.DoubleMatrix;

public class OrnsteinUhlenbeck extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams;
	public OrnsteinUhlenbeck(){
		numParams = 2;
	}
	public int getNumParams(){return numParams;}
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = Math.sqrt((x-xstar)*(x-xstar));
		double f = parameters.get(0)*parameters.get(0)*Math.exp(-d/(parameters.get(1)*parameters.get(1)));
		return f;

	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
//		System.out.println("OU: " + parameters.get(0) + " " + parameters.get(1) );

		double d = Math.sqrt((x.get(0)-xstar.get(0))*(x.get(0)-xstar.get(0)));
		double f = parameters.get(0)*parameters.get(0)*Math.exp(-d/(parameters.get(1)*parameters.get(1)));
		return f;
	}

	public DoubleMatrix computeSingleValue(DoubleMatrix parameters, DoubleMatrix X){
		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.rows+" instead of "+numParams);
		DoubleMatrix distance = squareDist(X);
		DoubleMatrix A = exp(distance.mul(1/(parameters.get(1)*parameters.get(1)))).mul(parameters.get(0)*parameters.get(0));
		return A;
	}

	public DoubleMatrix computeDerivatives(DoubleMatrix parameters, DoubleMatrix X, int index) {

		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.columns+" instead of "+numParams);
		if(index>numParams-1)
			throw new IllegalArgumentException("Wrong hyperparameters index "+index+" it should be smaller or equal to "+(numParams-1));

		DoubleMatrix A = new DoubleMatrix();
	
		DoubleMatrix distance = dist(X);
		if(index == 0){
			A = exp(distance.mul(-Math.sqrt(3)/parameters.get(2))).mmul(distance.mul(Math.sqrt(3)/parameters.get(1)).add(1));
		} else if(index == 1){
			A = exp(distance.mul(-Math.sqrt(3)/parameters.get(2))).mmul(distance.mul(-Math.sqrt(3)/parameters.get(1)*parameters.get(1))).mul(parameters.get(0)*parameters.get(0));
		} else {
			A = distance.mul(Math.sqrt(3)/parameters.get(2)*parameters.get(2)).add(distance.mmul(distance).mul(3/parameters.get(2)*parameters.get(2)*parameters.get(2))).mul(parameters.get(0)*parameters.get(0)).mmul(exp(distance.mul(-Math.sqrt(3)/parameters.get(2))));
		}
		
		return A;
	}
}
