package cov;
import org.jblas.DoubleMatrix;
public class SquaredExponential extends CovarianceFunction{
	public void train(){}
	public DoubleMatrix computeCovMatrix(DoubleMatrix k, DoubleMatrix kstar){return k;}
}
