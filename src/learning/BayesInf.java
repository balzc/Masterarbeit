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
	private DoubleMatrix aInv;
	private DoubleMatrix mean;
	private DoubleMatrix covar;
	private DoubleMatrix priorCovar;
	private DoubleMatrix priorMean;
	private double noiseVar;
	
	public BayesInf(DoubleMatrix x, DoubleMatrix y, DoubleMatrix priorcovar, DoubleMatrix priormean, double var){
		this.x = x;
		this.y = y;
		this.priorCovar = priorcovar;
		this.noiseVar = var;
		this.priorMean = priormean;
	
	}
	public void setup(){
//		System.out.println("setup");
//		Main.printMatrix(x);
//		Main.printMatrix(y);
		DoubleMatrix E = DoubleMatrix.eye(x.rows);
		DoubleMatrix priorCovarInv = Solve.solvePositive(priorCovar, E);
		
		
		a = x.mmul(x.transpose()).mul(1/noiseVar).add(priorCovarInv);
		aInv = Solve.solvePositive(a, E);
		DoubleMatrix sum = x.mmul(y).mul(1/noiseVar).add(priorCovarInv.mmul(priorMean));
		mean = aInv.mmul(sum);
//		Main.printMatrix(aInv);
		covar = aInv;
		
	}
	public DoubleMatrix getMean(){return mean;}
	public DoubleMatrix getCovar(){return covar;}
	public DoubleMatrix getA(){return a;}
	public DoubleMatrix getAInv(){return aInv;}

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
