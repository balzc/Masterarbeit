package cov;
import org.jblas.DoubleMatrix;
public class SquaredExponential extends CovarianceFunction{
	public DoubleMatrix parameters;
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x-xstar;
		return parameters.get(0)*parameters.get(0) *Math.exp(- 1/(2 * parameters.get(1)*parameters.get(1)) * d*d);
		
	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x.distance2(xstar);
		return parameters.get(0)*parameters.get(0) *Math.exp(- 1/(2 * parameters.get(1)*parameters.get(1)) * d*d);
	}
}
