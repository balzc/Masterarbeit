package learning;

import main.Main;

import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import cov.CovarianceFunction;

public class BayesInf {
	private DoubleMatrix prior;
	private DoubleMatrix likelihood;
	private DoubleMatrix parameters;
	private DoubleMatrix x;
	private DoubleMatrix y;
	private DoubleMatrix a;
	private DoubleMatrix mean;
	private DoubleMatrix covar;
	private DoubleMatrix priorCovar;

	private double var;
	
	public BayesInf(DoubleMatrix x, DoubleMatrix y, double var){
		this.x = x;
		this.y = y;
		this.var = var;
	
	}
	public void setup(){

		priorCovar = DoubleMatrix.eye(x.rows).mul(var);
		DoubleMatrix E = DoubleMatrix.eye(x.rows);
		DoubleMatrix priorCovarInv = Solve.solvePositive(priorCovar, E);
		a = x.mmul(x.transpose()).mul(1/var).add(priorCovarInv);
		DoubleMatrix aInv = Solve.solvePositive(a, E);
		mean = aInv.mmul(x).mmul(y).mul(1/var);

		covar = aInv;
		
	}
	public DoubleMatrix getMean(){return mean;}
	public DoubleMatrix getCovar(){return covar;}

	public DoubleMatrix generateSamples(int num, double small){

		DoubleMatrix k = DoubleMatrix.eye(num);
		DoubleMatrix smallId = DoubleMatrix.eye(k.columns).mul(small*small);
		k = k.add(smallId);
		DoubleMatrix l = Decompose.cholesky(k);
		DoubleMatrix u = DoubleMatrix.randn(k.columns);//DoubleMatrix.ones(k.columns);//
		DoubleMatrix y = l.transpose().mmul(u);

		return y;
	}
}
