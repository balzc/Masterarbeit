package cov;




import java.util.Arrays;

import org.jblas.DoubleMatrix;




public class CovSEiso extends CovarianceFunction{
	public DoubleMatrix parameters;
	public int numParams = 2;
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		double d = x-xstar;
		return parameters.get(0)*parameters.get(0) *Math.exp(- 1/(2 * parameters.get(1)*parameters.get(1)) * d*d);
		
	}
	
	   public DoubleMatrix compute(DoubleMatrix loghyper, DoubleMatrix X) {


	        double ell = Math.exp(loghyper.get(0,0));
	        double sf2 = Math.exp(2*loghyper.get(1,0));

	        DoubleMatrix K = exp(squareDist(X.transpose().mul(1/ell)).mul(-0.5)).mul(sf2);

	        return K;
	    }
	
	public DoubleMatrix computeCovarianceMatrix(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		   if(parameters.columns!=1 || parameters.rows!=numParams)
	            throw new IllegalArgumentException("Wrong number of hyperparameters, "+parameters.rows+" instead of "+numParams);

	        double ell = Math.exp(parameters.get(0,0));
	        double sf2 = Math.exp(2*parameters.get(1,0));



	        DoubleMatrix K = exp(squareDist(x.transpose().mul(1/ell),xstar.transpose().mul(1/ell)).mul(-0.5)).mul(sf2);

	        return K;
	}
	
	public DoubleMatrix computeDerivatives(DoubleMatrix loghyper, DoubleMatrix X, int index) {

        if(loghyper.columns!=1 || loghyper.rows!=numParams)
            throw new IllegalArgumentException("Wrong number of hyperparameters, "+loghyper.columns+" instead of "+numParams);
        if(index>numParams-1)
            throw new IllegalArgumentException("Wrong hyperparameters index "+index+" it should be smaller or equal to "+(numParams-1));

        double ell = Math.exp(loghyper.get(0,0));
        double sf2 = Math.exp(2*loghyper.get(1,0));
        DoubleMatrix tp = X.transpose().mmul(1/ell);
        DoubleMatrix tmp = squareDist(tp.transpose());
   
        DoubleMatrix A = null;
        if(index==0){
            A = exp(tmp.mul(-0.5)).mul(tmp).mul(sf2);
        } else {
            A = exp(tmp.mul(-0.5)).mul(2*sf2);
        }
        System.out.println("A: " + index);
        A.print();
        return A;
    }
	
	public static DoubleMatrix exp(DoubleMatrix A){

		DoubleMatrix out = new DoubleMatrix(A.rows,A.columns);
        for(int i=0; i<A.rows; i++)
            for(int j=0; j<A.columns; j++)
                out.put(i,j,Math.exp(A.get(i,j)));

        return out;
    }
	
	 private static DoubleMatrix squareDist(DoubleMatrix a){
	        return squareDist(a,a);
	    }

	    private static DoubleMatrix squareDist(DoubleMatrix a, DoubleMatrix b){
	    	DoubleMatrix C = new DoubleMatrix(a.columns,b.columns);
	        final int m = a.columns;
	        final int n = b.columns;
	        final int d = a.rows;

	        for (int i=0; i<m; i++){
	            for (int j=0; j<n; j++) {
	                double z = 0.0;
	                for (int k=0; k<d; k++) { double t = a.get(k,i) - b.get(k,j); z += t*t; }
	                C.put(i,j,z);
	            }
	        }

	        return C;
	    }
}
