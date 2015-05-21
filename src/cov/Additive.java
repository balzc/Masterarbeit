package cov;

import org.jblas.DoubleMatrix;

public class Additive extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams;
	public CovarianceFunction cov1;
	public CovarianceFunction cov2;
	
	public Additive(CovarianceFunction c1, CovarianceFunction c2){
		cov1 = c1;
		cov2 = c2;
		numParams = c1.getNumParams() + c2.getNumParams();
	}
	public int getNumParams(){return numParams;}

	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		
		DoubleMatrix p1 = DoubleMatrix.zeros(cov1.getNumParams());
		DoubleMatrix p2 = DoubleMatrix.zeros(cov2.getNumParams());

		for(int i = 0; i < cov1.getNumParams(); i++){
			p1.put(i, parameters.get(i));
		}
		for(int i = cov1.getNumParams(); i < getNumParams(); i++){
			p2.put(i - cov1.getNumParams(), parameters.get(i));
		}
		return cov1.computeCovariance(x, xstar, p1)+cov2.computeCovariance(x, xstar, p2);

	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		
		DoubleMatrix p1 = DoubleMatrix.zeros(cov1.getNumParams());
		DoubleMatrix p2 = DoubleMatrix.zeros(cov2.getNumParams());
	
		for(int i = 0; i < cov1.getNumParams(); i++){
			p1.put(i, parameters.get(i));
		}
		for(int i = cov1.getNumParams(); i < getNumParams(); i++){
			p2.put(i - cov1.getNumParams(), parameters.get(i));
		}

		return cov1.computeCovariance(x, xstar, p1)+cov2.computeCovariance(x, xstar, p2);
	}

	public DoubleMatrix computeSingleValue(DoubleMatrix parameters, DoubleMatrix X){
		DoubleMatrix p1 = DoubleMatrix.zeros(cov1.getNumParams());
		DoubleMatrix p2 = DoubleMatrix.zeros(cov2.getNumParams());
		for(int i = 0; i < cov1.getNumParams(); i++){
			p1.put(i, parameters.get(i));
		}
		for(int i = cov1.getNumParams(); i < getNumParams(); i++){
			p2.put(i - cov1.getNumParams(), parameters.get(i));
		}
		return cov1.computeSingleValue(p1,X).add(cov2.computeSingleValue(p2, X));
	}

	public DoubleMatrix computeDerivatives(DoubleMatrix parameters, DoubleMatrix X, int index) {

		if(parameters.columns!=1 || parameters.rows!=getNumParams())
			throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.columns+" instead of "+getNumParams());
		if(index>getNumParams()-1)
			throw new IllegalArgumentException("Wrong hyperparameters index "+index+" it should be smaller or equal to "+(getNumParams()-1));
		DoubleMatrix p1 = DoubleMatrix.zeros(cov1.getNumParams());
		DoubleMatrix p2 = DoubleMatrix.zeros(cov2.getNumParams());

		for(int i = 0; i < cov1.getNumParams(); i++){
			p1.put(i, parameters.get(i));
		}
		for(int i = cov1.getNumParams(); i < getNumParams(); i++){
			p2.put(i - cov1.getNumParams(), parameters.get(i));
		}
		if(index < cov1.getNumParams()){
			return cov1.computeDerivatives(p1, X, index).add(cov2.computeSingleValue(p2, X));
		} else{
			return cov2.computeDerivatives(p2, X, index-cov1.getNumParams()).add(cov1.computeSingleValue(p1, X));
		}
		
	}
}
