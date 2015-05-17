package cov;

import org.jblas.DoubleMatrix;

public abstract class CovarianceFunction {
	public DoubleMatrix parameters;
	public int numParams;
	public abstract int getNumParams();
	
	public double computeCovariance(double x, double xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		return x;
	}
	public double computeCovariance(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		this.parameters = parameters;
		return x.distance2(xstar);
	}
	public double computeCovariance(double x,  double[] parameters){
		
		return x;
	}
	public DoubleMatrix computeCovarianceMatrix(DoubleMatrix x, DoubleMatrix xstar, DoubleMatrix parameters){
		return x;
	}
	public double noise(int index1, int index2, double sn){
		return index1==index2 ? sn*sn : 0;
	}
	public DoubleMatrix computeDerivatives(DoubleMatrix loghyper, DoubleMatrix X, int index) {
		return loghyper;
	}
	
	public DoubleMatrix computeSingleValue(DoubleMatrix loghyper, DoubleMatrix X){
		return X;
	}
	
	public static DoubleMatrix exp(DoubleMatrix A){

		DoubleMatrix out = new DoubleMatrix(A.rows,A.columns);
		for(int i=0; i<A.rows; i++)
			for(int j=0; j<A.columns; j++)
				out.put(i,j,Math.exp(A.get(i,j)));

		return out;
	}

	public static DoubleMatrix squareDist(DoubleMatrix a){
		return squareDist(a,a);
	}

	public static DoubleMatrix squareDist(DoubleMatrix a, DoubleMatrix b){
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
	
	public static DoubleMatrix dist(DoubleMatrix a){
		return Dist(a,a);
	}
	public static DoubleMatrix Dist(DoubleMatrix a, DoubleMatrix b){
		DoubleMatrix C = new DoubleMatrix(a.columns,b.columns);
		final int m = a.columns;
		final int n = b.columns;
		final int d = a.rows;

		for (int i=0; i<m; i++){
			for (int j=0; j<n; j++) {
				double z = 0.0;
				for (int k=0; k<d; k++) { double t = a.get(k,i) - b.get(k,j); z += t; }
				C.put(i,j,z);
			}
		}

		return C;
	}
	
	public static DoubleMatrix sin(DoubleMatrix A){
		DoubleMatrix out = new DoubleMatrix(A.rows,A.columns);
		for(int i=0; i<A.rows; i++)
			for(int j=0; j<A.columns; j++)
				out.put(i,j,Math.sin(A.get(i,j)));
		return out;
	}
	public static DoubleMatrix cos(DoubleMatrix A){
		DoubleMatrix out = new DoubleMatrix(A.rows,A.columns);
		for(int i=0; i<A.rows; i++)
			for(int j=0; j<A.columns; j++)
				out.put(i,j,Math.cos(A.get(i,j)));
		return out;
	}
}
