package cov;

import org.jblas.DoubleMatrix;
// p0*(1+(d*sqrt(3)/p1))*exp(-(d*sqrt(3)/p2)
public class Matern extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams;
	public Matern(){
		numParams = 3;
	}
	public int getNumParams(){return numParams;}
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x-xstar;
		double c = Math.exp((-d*Math.sqrt(3))/parameters.get(2));
		double f = parameters.get(0)*(1+(d*Math.sqrt(3))/parameters.get(1))*c;
		return f;

	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x.get(0)-xstar.get(0);
		double c = Math.exp((-d*Math.sqrt(3))/parameters.get(2));
		double f = parameters.get(0)*(1+(d*Math.sqrt(3))/parameters.get(1))*c;
		return f;
	}

	public DoubleMatrix computeSingleValue(DoubleMatrix parameters, DoubleMatrix X){
		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.rows+" instead of "+numParams);
		DoubleMatrix distance = dist(X);
		DoubleMatrix A = exp(distance.mul(Math.sqrt(3)/parameters.get(2))).mmul(distance.mul(Math.sqrt(3)/parameters.get(1)).add(1)).mul(parameters.get(0));
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
			A = exp(distance.mul(-Math.sqrt(3)/parameters.get(2))).mmul(distance.mul(-Math.sqrt(3)/parameters.get(1)*parameters.get(1))).mul(parameters.get(0));
		} else {
			A = distance.mul(Math.sqrt(3)/parameters.get(2)*parameters.get(2)).add(distance.mmul(distance).mul(3/parameters.get(2)*parameters.get(2)*parameters.get(2))).mul(parameters.get(0)).mmul(exp(distance.mul(-Math.sqrt(3)/parameters.get(2))));
		}
		
		return A;
	}
}
