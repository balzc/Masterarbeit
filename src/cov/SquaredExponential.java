package cov;



import org.jblas.DoubleMatrix;


public class SquaredExponential extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams;
	public SquaredExponential(){
		numParams = 2;
	}
	public int getNumParams(){return numParams;}
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = (x-xstar)*(x-xstar);
		return parameters.get(0)*parameters.get(0) *Math.exp(- 1/(2 * parameters.get(1)*parameters.get(1)) * d*d);

	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x.distance2(xstar);
		return parameters.get(0)*parameters.get(0) *Math.exp(- 1/(2 * parameters.get(1)*parameters.get(1)) * d*d);
	}

	public DoubleMatrix computeSingleValue(DoubleMatrix loghyper, DoubleMatrix X){
		if(loghyper.columns!=1 || loghyper.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+loghyper.rows+" instead of "+numParams);

		double ell = loghyper.get(0,0);
		double sf2 = loghyper.get(1,0);

		DoubleMatrix K = exp(squareDist(X.transpose().mmul(1/ell)).mmul(-0.5)).mmul(sf2);
		K = K.mul(K);
		return K;
	}

	public DoubleMatrix computeDerivatives(DoubleMatrix parameters, DoubleMatrix X, int index) {

		if(parameters.columns!=1 || parameters.rows!=numParams)
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.columns+" instead of "+numParams);
		if(index>numParams-1)
			throw new IllegalArgumentException("Wrong hyperparameters index "+index+" it should be smaller or equal to "+(numParams-1));

		double ell = parameters.get(0,0);
		double sf2 = parameters.get(1,0);
		DoubleMatrix tp = X.transpose().mul(1/ell);
		DoubleMatrix tmp = squareDist(tp.transpose());



		DoubleMatrix A = null;
		if(index==0){
			A = exp(tmp.mmul(-0.5)).mul(tmp).mmul(sf2);
		} else {
			A = exp(tmp.mmul(-0.5)).mmul(2*sf2);
		}

		
		return A;
	}
	
	
}
