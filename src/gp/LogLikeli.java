package gp;



import org.jblas.Decompose;
import org.jblas.DoubleMatrix;

import cov.*;

import cobyla.*;


//marginal likelihood using the squared exponential covariance function
//has 3 parameters: sf, ell, sn
public class LogLikeli implements Calcfc{

	private static final double tol = 0.001;
	
	private static boolean DEBUG = false;

	//Dxn vector of input data on x-axis
	private DoubleMatrix dataX;
	//nx1 vector of input data on y-axis
	private DoubleMatrix dataY;
	private CovarianceFunction cf;
	private DoubleMatrix distances;
	
	//hacky way of getting noise
	private double sn;
	private boolean noiseKnown;

	public LogLikeli(DoubleMatrix x, DoubleMatrix y, CovarianceFunction cf, boolean noise, double sn){
		this.dataX = x;
		this.dataY = y;
		this.cf = cf;
		this.noiseKnown = noise;
		this.sn = sn;
	}
	
	public LogLikeli(DoubleMatrix x, DoubleMatrix y, CovarianceFunction cf, boolean noise){
		this.dataX = x;
		this.dataY = y;
		this.cf = cf;
		this.noiseKnown = noise;
	}

	public void computeDistances(){
		final int numData = dataX.columns;
		distances = new DoubleMatrix(numData,numData);
		for(int i = 0; i<numData; i++){
			for(int j = i; j<numData; j++){
				double d; 

				d = dataX.getColumn(i).distance2(dataX.getColumn(j)); 

				distances.put(i,j, d);
				distances.put(j,i, d);
			}
		}
	}
	
	/**
     * The objective and constraints function evaluation method used in COBYLA2 minimization.
     * @param n Number of variables.
     * @param m Number of constraints.
     * @param x Variable values to be employed in function and constraints calculation.
     * @param con Calculated function values of the constraints.
     * @return Calculated objective function value.
     */
	@Override
	public double Compute(int n, int m, double[] theta, double[] con) {
		for(int i = 0; i<n; i++){
			con[2*i] = theta[i]>0? theta[i]:0.01;
			con[2*i+1] = Double.MAX_VALUE;
			/*if(theta.length>3 && i == 6){
				con[2*i+1] = theta[i]<1? theta[i] : .9;
			}*/
		}
		return evaluate(theta);
	}
	
	//order of params: sf,ell,sn
	public double evaluate(double[] theta) {
		
		//System.out.println("Paramas");
		for(int i = 0; i<theta.length; i++){
			//System.out.println(theta[i]);
			if(theta[i]<tol) theta[i]=0.01;
		}
		if(theta.length == 7 &&theta[5]>1) theta[5]=1;
		//if(theta.length == 8 &&theta[6]>1) theta[6]=1;
		if(theta.length == 9 &&theta[7]>1) theta[7]=1;
		
		
		/*System.out.println("Paramas");
		for(int i = 0; i<theta.length; i++){
			System.out.println(theta[i]);
		
		}*/
		
		final int numData = dataX.columns;
		DoubleMatrix K = new DoubleMatrix(numData,numData);
		//double par[] = new double[theta.length-1];

		
		//1. generate covariance matrix K with parameters theta
		double res;
		for(int i = 0; i<numData; i++){
			for(int j = i; j<numData; j++){
				if (noiseKnown){
					//res = cf.evaluateParam(dataX.getColumn(i), dataX.getColumn(j), theta)+cf.noise(i, j, sn);

					res = cf.computeCovariance(distances.get(i,j),theta)+cf.noise(i, j, sn);

					K.put(i,j, res);
					K.put(j,i, res);
				}
				else{
					
					//res = cf.evaluateParam(dataX.getColumn(i), dataX.getColumn(j), theta)+cf.noise(i, j, theta[theta.length-1]);
				
					res = cf.computeCovariance(distances.get(i,j),theta)+cf.noise(i, j, theta[theta.length-1]);

					K.put(i,j, res);
					K.put(j,i, res);
				}
				
			}

		}
		/*
		System.out.println("K");
		K.print();
		System.out.println("KK");
		KK.print();*/
		//K.print();
		if(DEBUG){
			K.print();
		}
		//2. compute alpha = K^-1 * y
		DoubleMatrix U = Decompose.cholesky(K);
		//2. compute
		DoubleMatrix alpha = forwardSubstitution(U.transpose(), dataY);
		alpha = backwardSubstitution(U, alpha);

		if (DEBUG){
			System.out.println("Alpha");
			alpha.print();
		}
		//3. compute log(det(K)) = sum_i(log(U_ii))
		double logDet = 0;
		for (int i = 0; i< K.columns; i++){
			logDet += Math.log(U.get(i,i));
		}
		if (DEBUG){
			System.out.println("logdet = "+logDet);
		}
		//4. compute log-likelihood
		//log( p(y|X,theta) = -0.5* y'*alpha - log(det(K)) - n/2*log(2pi)
		if(DEBUG){
			System.out.println("Erster term"+-0.5* dataY.dot(alpha));
		}
		double L =  -0.5* dataY.dot(alpha) - logDet - 0.5*numData*Math.log(2*Math.PI);
		return -L;

	}




	//solve the equation Ax = b by forward substitution (A must be lower triangular matrix)
	public static DoubleMatrix forwardSubstitution(DoubleMatrix A, DoubleMatrix b){
		DoubleMatrix res = new DoubleMatrix(b.length);
		for (int i = 0; i< b.length; i++){
			res.put(i, b.get(i));
			for(int j = 0; j<i; j++){
				res.put(i, res.get(i)-A.get(i,j)*res.get(j));
			}
			res.put(i, res.get(i)/A.get(i,i));
		}
		return res;
	}

	//solve the equation Ax = b by forward substitution (A must be upper triangular matrix)
	public static DoubleMatrix backwardSubstitution(DoubleMatrix A, DoubleMatrix b){
		DoubleMatrix res = new DoubleMatrix(b.length);
		for (int i = b.length-1; i>= 0; i--){
			res.put(i, b.get(i));
			for(int j = i+1; j<b.length; j++){
				res.put(i, res.get(i)-A.get(i,j)*res.get(j));
			}
			res.put(i, res.get(i)/A.get(i,i));
		}
		return res;
	}


	public DoubleMatrix getDistances(){
		return this.distances;
	}
	

}

