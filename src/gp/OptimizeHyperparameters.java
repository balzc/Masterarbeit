package gp;

import org.jblas.DoubleMatrix;

import cov.CovarianceFunction;


import cobyla.Cobyla;

public class OptimizeHyperparameters {

	private static double rhobeg = .5;
	private static double rhoend = 1.0e-4;
	private static int iprint = 0;
	private static int maxfun = 100;

	//profiling
	private static double time;
	private static boolean PROFILING = true;
	private static int numRep = 100;
	

	/**
	 * 
	 * @param dataX
	 * @param dataY
	 * @param cf
	 * @param numVar
	 * @return res[][]: res[0][0] = max. function value, res[1][] = array of optimal parameter values in that order (if parameter exists): sf, ell, sn, rho
	 */
	public static double[][] optimizeParams(DoubleMatrix dataX, DoubleMatrix dataY, CovarianceFunction cf, int numVar,boolean noiseKnown, double sn){
		if(PROFILING){
			time = System.nanoTime();
		}

		LogLikeli lf = new LogLikeli(dataX,dataY,cf,noiseKnown,sn);
		lf.computeDistances();

		int numConstr = 2*numVar;
		double[] startX = new double[numVar];

		double[][] opt = new double[numVar][numVar];
		double maxLoglikeli = Double.NEGATIVE_INFINITY;
		double[][] res = null;
		
		for (int r = 0; r<numRep; r++){
			for (int i = 0; i<numVar; i++){
				startX[i] = Math.random()*1.;
				
			}
			if(numVar==8){
				startX[6] = Math.random();
			}


			res =  Cobyla.FindMinimum(lf, numVar, numConstr, startX, rhobeg, rhoend, iprint, maxfun);
			if(res[0][0]>maxLoglikeli){
				maxLoglikeli = res[0][0];
				opt[0][0] = res[0][0];
			}
			/*
			System.out.println("loglikeli:"+res[0][0]);
			for(int i = 0; i<res[1].length; i++){
				System.out.println(res[1][i]);
			}*/
		}

		if(PROFILING){
			time = System.nanoTime()-time;
			System.out.println("@optimizeParams(): Time elapsed: "+ time/Math.pow(10,9));
		}
		return res;
	}


	public static double getRhobeg() {
		return rhobeg;
	}


	public static void setRhobeg(double rhobeg) {
		OptimizeHyperparameters.rhobeg = rhobeg;
	}


	public static double getRhoend() {
		return rhoend;
	}


	public static void setRhoend(double rhoend) {
		OptimizeHyperparameters.rhoend = rhoend;
	}


	public static int getMaxfun() {
		return maxfun;
	}


	public static void setMaxfun(int maxfun) {
		OptimizeHyperparameters.maxfun = maxfun;
	}


	public static int getNumRep() {
		return numRep;
	}


	public static void setNumRep(int numRep) {
		OptimizeHyperparameters.numRep = numRep;
	}


	public static boolean isPROFILING() {
		return PROFILING;
	}


	public static void setPROFILING(boolean PROFILING) {
		PROFILING = PROFILING;
	}
	
	
	
}
