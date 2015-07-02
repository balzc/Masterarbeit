package gp;


import cov.CovarianceFunction;
import cov.SquaredExponential;
import main.Main;

import org.jblas.Decompose;
import org.jblas.Solve;
import org.jblas.DoubleMatrix;

import com.sun.org.apache.bcel.internal.util.SecuritySupport;

import util.FileHandler;
public class GP {
	private CovarianceFunction covf;
	private DoubleMatrix trainIn;
	private DoubleMatrix trainOut;
	private DoubleMatrix testIn;
	private DoubleMatrix testOut;
	
	private DoubleMatrix l;
	private DoubleMatrix alpha;
	private DoubleMatrix targets;
	private double noiselevel;
	private DoubleMatrix trainCov;
	private DoubleMatrix testCov;
	private DoubleMatrix testTrainCov;
	private DoubleMatrix predMean;


	private DoubleMatrix predVar;
	private double logLikelihood;
	private DoubleMatrix parameters;
	private int numTest;
	private int numTrain;
	
	public GP(DoubleMatrix trainInput, DoubleMatrix testInput, DoubleMatrix parameters, CovarianceFunction cf, double noisel){
		this.noiselevel = noisel;
		this.trainIn = trainInput;
		this.testIn = testInput;
		this.covf = cf;
		this.numTest = this.testIn.columns;
		this.numTrain = this.trainIn.columns;
		this.parameters = parameters;
	
//		
//		
//		DoubleMatrix beta = Solve.pinv(l).mmul(Solve.pinv(l).mmul(trainOut.transpose()));
//		alpha = Solve.solve(l.transpose(), Solve.solve(l, trainOut.transpose()));
//		predMean = testCov.transpose().mmul(alpha);
//		predVar = Solve.solve(l, testCov);
//		System.out.println("End");

		
	}
	
	public void setup(DoubleMatrix trainOut){
		this.trainOut = trainOut;
		trainCov = computeCovMatrix(trainIn, trainIn, parameters);
		testCov = computeCovMatrix(testIn, testIn, parameters);
		testTrainCov = computeCovMatrix(trainIn, testIn, parameters);
//		l = computeL();
//		alpha = computeAlpha();
		predMean = computeMean2();
		predVar = computeVariance();
	}
	
	
	
	
	public DoubleMatrix computeAlpha(){
		DoubleMatrix cova = computeCovMatrix(trainIn, trainIn,parameters);
		DoubleMatrix identity = DoubleMatrix.eye(trainIn.rows);
		DoubleMatrix temp = cova.add(identity.mul(noiselevel));
		DoubleMatrix el = Decompose.cholesky(temp);
		return Solve.solve(el.transpose(), Solve.solve(el, trainOut));
	}
	
	public DoubleMatrix computeL(){
		DoubleMatrix identity = DoubleMatrix.eye(trainCov.rows);
		DoubleMatrix temp = trainCov.add(identity.mul(noiselevel));
		DoubleMatrix el = Decompose.cholesky(temp);
		return el;
	}
	

	public DoubleMatrix computeMean(){
		DoubleMatrix cova = computeCovMatrix(trainIn, testIn, parameters);
		DoubleMatrix identity = DoubleMatrix.eye(trainCov.rows);
		DoubleMatrix temp = trainCov.add(identity.mul(noiselevel));
		System.out.println("Temp1");

		Main.printMatrix(temp);
		DoubleMatrix covInv = Solve.solvePositive(temp, identity);
		System.out.println("Alpha1");

		Main.printMatrix(covInv.mmul(trainOut));
		DoubleMatrix mean = cova.transpose().mmul(covInv).mmul(trainOut);//cova.transpose().mmul(alpha);
		return mean;
	}
	
	public DoubleMatrix computeMean2(){
		DoubleMatrix cova = computeCovMatrix(trainIn, testIn, parameters);
		DoubleMatrix identity = DoubleMatrix.eye(trainCov.rows);
		DoubleMatrix temp = trainCov.add(identity.mul(noiselevel));
//		System.out.println("Temp2");
//
//		Main.printMatrix(temp);
		DoubleMatrix l = Decompose.cholesky(temp).transpose();


		DoubleMatrix alpha = Solve.solve( l.transpose(),Solve.solve(l, trainOut));
//		System.out.println("Alpha2");
//
//		Main.printMatrix(alpha);
		DoubleMatrix mean = cova.transpose().mmul(alpha);
		return mean;
	}
	
	public DoubleMatrix computeVariance(){
		DoubleMatrix identity = DoubleMatrix.eye(trainCov.rows);
		DoubleMatrix temp = trainCov.add(identity.mul(noiselevel));
		DoubleMatrix l = Decompose.cholesky(temp).transpose();
		DoubleMatrix v = Solve.solve(l,testTrainCov);
		
	
		DoubleMatrix variance = testCov.sub(v.transpose().mmul(v));
		return variance;
	}

//	public double negativeLogLikelihood2(DoubleMatrix parameters, DoubleMatrix in, DoubleMatrix out, DoubleMatrix df0){
//		DoubleMatrix alpha = computeAlpha(in, out, parameters, noiselevel);
//		double loglike = out.transpose().mmul(alpha).get(0,0)*(-0.5);
//		DoubleMatrix cova = computeCovMatrix(in, in, parameters);
//		DoubleMatrix identity = DoubleMatrix.eye(in.rows);
//		DoubleMatrix temp = cova.add(identity.mul(noiselevel));
//		DoubleMatrix el = Decompose.cholesky(temp);
//		for(int i = 0; i < el.rows; i++){
//			loglike -= Math.log(el.get(i, i));
//		}
//		loglike -= (numTrain/2)*Math.log(2*Math.PI);
//		
//	    DoubleMatrix W = bSubstitutionWithTranspose(el,(fSubstitution(el,DoubleMatrix.eye(in.columns)))).sub(alpha.mmul(alpha.transpose()));     // precompute for convenience
//
//        for(int i=0; i<df0.rows; i++){
//            df0.put(i,0,W.mul(covf.computeDerivatives(parameters, in, i)).sum()/2);
//        }
//        df0.print();
//		return -loglike;
//	}
	public double negativeLogLikelihood(DoubleMatrix logtheta,DoubleMatrix x, DoubleMatrix y, DoubleMatrix df0) {

        int n = 1;// x.rows;
        
        DoubleMatrix K = covf.computeSingleValue(logtheta, x);    // compute training set covariance matrix
        DoubleMatrix cd = Decompose.cholesky(K);
        if(!true) {
            throw new RuntimeException("The covariance Matrix is not SDP, check your covariance function (maybe you mess the noise term..)");
        }   else{
        	DoubleMatrix L = cd.transpose();                // cholesky factorization of the covariance
        
            DoubleMatrix alpha = bSubstitutionWithTranspose(L,fSubstitution(L,y.transpose()));
            

            // compute the negative log marginal likelihood
            double lml = (y.mmul(alpha).mmul(0.5)).get(0,0);

            for(int i=0; i<L.rows; i++) lml+=Math.log(L.get(i,i));
            lml += 0.5*n*Math.log(2*Math.PI);




            DoubleMatrix W = bSubstitutionWithTranspose(L,(fSubstitution(L,DoubleMatrix.eye(n)))).sub(alpha.mmul(alpha.transpose()));     // precompute for convenience
        
            for(int i=0; i<df0.rows; i++){
            	DoubleMatrix derivatives = covf.computeDerivatives(logtheta, x, i);
            	df0.put(i,0,(W.mul(derivatives)).sum()/2);
            }
//            df0.print();
//            System.out.println(lml);
            return -lml;
        }
    }
	// compute the covariance matrix of two vector inputs TODO: double computations unnecessary when k = kstar
	public DoubleMatrix computeCovMatrix(DoubleMatrix k, DoubleMatrix kstar, DoubleMatrix parameters){
		DoubleMatrix result = new DoubleMatrix(k.rows,kstar.rows);
		for(int i = 0; i<k.rows; i++){
			for(int j = 0; j<kstar.rows; j++){
//				System.out.println("i: " + i + " j: " + j);
//				k.getRow(i).print();
//				kstar.getRow(j).print();
				result.put(i, j, covf.computeCovariance(k.getRow(i), kstar.getRow(j), parameters));
			}
		}
		return result;
	}

	
	public DoubleMatrix generateSamples(DoubleMatrix in, DoubleMatrix parameters, double small, CovarianceFunction covf){

		DoubleMatrix k = computeCovMatrix(in, in, parameters);
		DoubleMatrix smallId = DoubleMatrix.eye(k.columns).mul(small*small);
		k = k.add(smallId);
		DoubleMatrix l = Decompose.cholesky(k);
		DoubleMatrix u = DoubleMatrix.randn(k.columns);//DoubleMatrix.ones(k.columns);//
		DoubleMatrix y = l.transpose().mmul(u);

		return y;
	}
	
    private final static double INT = 0.1;                // don't reevaluate within 0.1 of the limit of the current bracket

    private final static double EXT = 6.0;                // extrapolate maximum 3 times the current step-size

    private final static int MAX = 20;                    // max 20 function evaluations per line search

    private final static double RATIO = 10;               // maximum allowed slope ratio

    private final static double SIG = 0.9, RHO = SIG/2;   // SIG and RHO are the constants controlling the Wolfe-
    /* Powell conditions. SIG is the maximum allowed absolute ratio between
    * previous and new slopes (derivatives in the search direction), thus setting
    * SIG to low (positive) values forces higher precision in the line-searches.
    * RHO is the minimum allowed fraction of the expected (from the slope at the
    * initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    * Tuning of SIG (depending on the nature of the function to be optimized) may
    * speed up the minimization; it is probably not worth playing much with RHO.

    /* This function is part of the jgpml Project.
     * http://github.com/renzodenardi/jgpml
     *
     * Copyright (c) 2011 Renzo De Nardi and Hugo Gravato-Marques
     *
     * Permission is hereby granted, free of charge, to any person
     * obtaining a copy of this software and associated documentation
     * files (the "Software"), to deal in the Software without
     * restriction, including without limitation the rights to use,
     * copy, modify, merge, publish, distribute, sublicense, and/or sell
     * copies of the Software, and to permit persons to whom the
     * Software is furnished to do so, subject to the following
     * conditions:
     *
     * The above copyright notice and this permission notice shall be
     * included in all copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
     * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
     * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
     * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
     * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
     * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
     * OTHER DEALINGS IN THE SOFTWARE.
     */
    public DoubleMatrix minimize(DoubleMatrix params, int length, DoubleMatrix in, DoubleMatrix out){

        double A, B;
        double x1, x2, x3, x4;
        double f0, f1, f2, f3, f4;
        double d0, d1, d2, d3, d4;
        DoubleMatrix df0, df3;
        DoubleMatrix fX;

        double red = 1.0;

        int i = 0;
        int ls_failed = 0;

        int sizeX = params.rows;

        df0 = new DoubleMatrix(sizeX,1);
        f0 = negativeLogLikelihood(params, in, out,df0);

//        System.out.println("f0:" + f0);
        //f0 = f.evaluate(params,cf, in, out, df0);
        fX = new DoubleMatrix(new double[]{f0});

        i = (length < 0) ? i+1 : i;
        DoubleMatrix s = df0.mmul(-1);

        // initial search direction (steepest) and slope
        d0 = s.mmul(-1).transpose().mmul(s).get(0,0);
        x3 = red/(1-d0);                                  // initial step is red/(|s|+1)

        final int nCycles = Math.abs(length);

        int success;

        double M;
        while (i < nCycles){
            //System.out.println("-");
            i = (length > 0) ? i+1 : i;    // count iterations?!


            // make a copy of current values
            double F0 = f0;
            DoubleMatrix X0 = params.dup();
            DoubleMatrix dF0 = df0.dup();

            M = (length>0) ? MAX : Math.min(MAX, -length-i);

            while (true) {                            // keep extrapolating as long as necessary

                x2 = 0;
                f2 = f0;
                d2 = d0;
                f3 = f0;
                df3 = df0.dup();

                success = 0;

                while (success == 0 && M > 0){
                    //try
                    M = M - 1;   i = (length < 0) ? i+1 : i;    // count iterations?!

                    DoubleMatrix m1 = params.add(s.mmul(x3));
                    //f3 = f.evaluate(m1,cf, in, out, df3);

                    f3 = negativeLogLikelihood(m1, in, out,df3);
//                    System.out.println("start small");
//
//                    System.out.println("F0:" + F0);
//                    System.out.println("df3:");
//                    df3.print();
//                    System.out.println("m1:");
//                    m1.print();
//                    System.out.println("f3:" + f3);
//                    System.out.println("end");

                    if (Double.isNaN(f3) || Double.isInfinite(f3) || hasInvalidNumbers(df3.toArray())){
                        x3 = (x2+x3)/2;     // catch any error which occured in f
                    }else{
                        success = 1;
                    }

                }

                if (f3 < F0){                   // keep best values
                    X0 = s.mmul(x3).add(params);
                    F0 = f3;
                    dF0 = df3;
//                    System.out.println("found part 1");
//                    X0.print();
                }

                d3 = df3.transpose().mmul(s).get(0,0);  // new slope

                if (d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0){  // are we done extrapolating?
                    break;
                }

                x1 = x2; f1 = f2; d1 = d2;                   // move point 2 to point 1
                x2 = x3; f2 = f3; d2 = d3;                  // move point 3 to point 2

                A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);     // make cubic extrapolation
                B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);

                x3 = x1-d1*(x2-x1)*(x2-x1)/(B + Math.sqrt(B*B-A*d1*(x2-x1)));  // num. error possible, ok!

                if (Double.isNaN(x3) || Double.isInfinite(x3) || x3 < 0)     // num prob | wrong sign?
                    x3 = x2*EXT;                             // extrapolate maximum amount
                else if (x3 > x2*EXT)                        // new point beyond extrapolation limit?
                    x3 = x2*EXT;                            // extrapolate maximum amount
                else if (x3 < x2+INT*(x2-x1))               // new point too close to previous point?
                    x3 = x2+INT*(x2-x1);

            }

            f4 = 0;
            x4 = 0;
            d4 = 0;

            while ((Math.abs(d3) > -SIG*d0 ||
                    f3 > f0+x3*RHO*d0) && M > 0){               // keep interpolating

                if (d3 > 0 || f3 > f0+x3*RHO*d0){                // choose subinterval
                    x4 = x3; f4 = f3; d4 = d3;                  // move point 3 to point 4
                }else{
                    x2 = x3; f2 = f3; d2 = d3;                          // move point 3 to point 2
                }

                if (f4 > f0){
                    x3 = x2-(0.5*d2*(x4-x2)*(x4-x2))/(f4-f2-d2*(x4-x2));    // quadratic interpolation
                }else{
                    A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                        // cubic interpolation
                    B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
                    x3 = x2+(Math.sqrt(B*B-A*d2*(x4-x2)*(x4-x2))-B)/A;      // num. error possible, ok!
                }

                if (Double.isNaN(x3) || Double.isInfinite(x3)){
                    x3 = (x2+x4)/2;               // if we had a numerical problem then bisect
                }

                x3 = Math.max(Math.min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  // don't accept too close

                DoubleMatrix m1 = s.mmul(x3).add(params);
                //f3 = f.evaluate(m1,cf, in, out, df3);
                
                f3 = negativeLogLikelihood(m1, in, out,df3);
//                System.out.println("start");
//
//                System.out.println("df3:");
//                df3.print();
//                System.out.println("m1:");
//                m1.print();
//                System.out.println("F0:" + F0);
//
//                System.out.println("f3:" + f3);
//                System.out.println("end");

                if (f3 < F0){
                    X0 = m1.dup();
                    F0 = f3;
                    dF0 = df3.dup(); 
//                    System.out.println("found part 2");
//                    X0.print();
				// keep best values
                }

                M = M - 1;  i = (length < 0) ? i+1 : i;          // count iterations?!

                d3 = df3.transpose().mmul(s).get(0,0); // new slope

            }                                                    // end interpolation

            if (Math.abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0){     // if line search succeeded
                params = s.mmul(x3).add(params);
                f0 = f3;

                double[] elem = fX.toArray();
                double[] newfX = new double[elem.length + 1];

                System.arraycopy(elem, 0, newfX, 0, elem.length);
                newfX[elem.length-1] = f0;
                fX = new DoubleMatrix(newfX);                 // update variables


//                System.out.println("Function evaluation "+i+" Value "+f0);

                
                double tmp1 = df3.transpose().mmul(df3).sub(df0.transpose().mmul(df3)).get(0,0);
                double tmp2 = df0.transpose().mmul(df0).get(0,0);

                s =  s.mmul(tmp1/tmp2).sub(df3);

                df0 = df3;                          // swap derivatives
                d3 = d0;
                d0 = df0.transpose().mmul(s).get(0,0);

                if (d0 > 0){                        // new slope must be negative
                    s = df0.mmul(-1);              // otherwise use steepest direction
                    d0 = s.mmul(-1).transpose().mmul(s).get(0,0);
                }

                x3 = x3 * Math.min(RATIO, d3/(d0-Double.MIN_VALUE));    // slope ratio but max RATIO
                ls_failed = 0;                                          // this line search did not fail

            }else{

                params = X0; f0 = F0; df0 = dF0;                     // restore best point so far

                if (ls_failed == 1 || i > Math.abs(length)){    // line search failed twice in a row
                    break;                                      // or we ran out of time, so we give up
                }

                s = df0.mmul(-1); d0 = s.mmul(-1).transpose().mmul(s).get(0,0);      // try steepest
                x3 = 1/(1-d0);
                ls_failed = 1;                                                     // this line search failed

            }
        }

        return params;
    }
    
    private static boolean hasInvalidNumbers(double[] array){

        for(double a : array){
            if(Double.isInfinite(a) || Double.isNaN(a)){
                return true;
            }
        }

        return false;
    }
    
    private static DoubleMatrix fSubstitution(DoubleMatrix L, DoubleMatrix B){

        final double[][] l = L.toArray2();
        final double[][] b = B.toArray2();
        final double[][] x = new double[B.rows][B.columns];

        final int n = x.length;

        for(int i=0; i<B.columns; i++){
            for(int k=0; k<n; k++){
                x[k][i] = b[k][i];
                for(int j=0; j<k; j++){
                    x[k][i] -= l[k][j] * x[j][i];
                }
                x[k][i] /= l[k][k];
            }
        }
        return new DoubleMatrix(x);
    }



    private static DoubleMatrix bSubstitution(DoubleMatrix L, DoubleMatrix B){

        final double[][] l = L.toArray2();
        final double[][] b = B.toArray2();
        final double[][] x = new double[B.rows][B.columns];

        final int n = x.length-1;

        for(int i=0; i<B.columns; i++){
            for(int k=n; k > -1; k--){
                x[k][i] = b[k][i];
                for(int j=n; j>k; j--){
                    x[k][i] -= l[k][j] * x[j][i];
                }
                x[k][i] /= l[k][k];
            }
        }
        return new DoubleMatrix(x);

    }

     private static DoubleMatrix bSubstitutionWithTranspose(DoubleMatrix L, DoubleMatrix B){

        final double[][] l = L.toArray2();
        final double[][] b = B.toArray2();
        final double[][] x = new double[B.rows][B.columns];

        final int n = x.length-1;

        for(int i=0; i<B.columns; i++){
            for(int k=n; k > -1; k--){
                x[k][i] = b[k][i];
                for(int j=n; j>k; j--){
                    x[k][i] -= l[j][k] * x[j][i];
                }
                x[k][i] /= l[k][k];
            }
        }
        return new DoubleMatrix(x);

    }
     
     public void test(){
 		int noData = 1000;
 		double[] dataX = new double[noData];
 		double[] dataY =  new double[noData];
 		for(int i=0; i< dataX.length; i++){
 			dataX[i] = i;
 			dataY[i] = Math.random();
 		}
 		double[] dataP = {0.5,0.5};
 		double[] dataTest = {11,12,13,14,15,16};
 		double nl = 0;
 		noiselevel = nl;
 		DoubleMatrix X = new DoubleMatrix(dataX);
 		DoubleMatrix Y = new DoubleMatrix(dataY);
 		DoubleMatrix P = new DoubleMatrix(dataP);
// 		DoubleMatrix testIn = new DoubleMatrix(dataTest);
// 		DoubleMatrix co = computeCovMatrix(X, X, P);
// 		co.print();
 		DoubleMatrix samples = generateSamples(X, P, nl, covf);
// 		String dest = "/Users/Balz/Downloads/test.csv";
// 		samples.print();
// 		FileHandler.matrixToCsv(samples, dest);
// 		DoubleMatrix tsamples = FileHandler.csvToMatrix(dest);
// 		samples.print();
 		System.out.print("[");
 		for(int i = 0; i<noData; i++){
 			System.out.print(Y.get(i)+ "; ");

 		}
 		System.out.println("]");
 		System.out.print("[");

 		for(int i = 0; i<noData; i++){
 			System.out.print(X.get(i)+ "; ");

 		}
 		System.out.println("]");
 		System.out.print("[");

 		for(int i = 0; i<noData; i++){
 			System.out.print(samples.get(i)+ "; ");

 		}
 		System.out.println("]");
 		DoubleMatrix params = minimize(P, -100, X, samples);
 		params.print();
// 		samples.print();
// 		int noruns = 20;
// 		DoubleMatrix[] params = new DoubleMatrix[noruns];
// 		double maxloglike = -9999999;
// 		int maxrun = 0;
// 		double[] loglikelies = new double[noruns];
// 		for(int i = 0; i < noruns; i++){
// 			double p1 = Math.random();
// 			double p2 = Math.random();
// 			DoubleMatrix ps = new DoubleMatrix(new double[] {p1,p2});
// 			System.out.println("run: " + i);
// 			ps.print();
// 			params[i] = minimize(ps,100,X, samples);
// 			params[i].print();
// 			loglikelies[i] = negativeLogLikelihood(params[i], X, samples, new DoubleMatrix(new double[] {1,2}));
// 			if(loglikelies[i] > maxloglike){
// 				maxloglike = loglikelies[i];
// 				maxrun = i;
// 			}
// 		}
// 		System.out.println(loglikelies[maxrun]);
// 		params[maxrun].print();
// 		double[][] param = OptimizeHyperparameters.optimizeParams(X.transpose(), samples.transpose(), covf, 2, false, nl);
// 		System.out.println(param[0][0]);
// 		System.out.println(param[0][1]);

 	}
     
     
 	public DoubleMatrix getPredMean() {
		return predMean;
	}

	public void setPredMean(DoubleMatrix predMean) {
		this.predMean = predMean;
	}

	public DoubleMatrix getPredVar() {
		return predVar;
	}

	public void setPredVar(DoubleMatrix predVar) {
		this.predVar = predVar;
	}
	
	public DoubleMatrix getL() {
		return l;
	}

	public void setL(DoubleMatrix l) {
		this.l = l;
	}

	public DoubleMatrix getAlpha() {
		return alpha;
	}

	public void setAlpha(DoubleMatrix alpha) {
		this.alpha = alpha;
	}

	public DoubleMatrix getTrainCov() {
		return trainCov;
	}

	public void setTrainCov(DoubleMatrix trainCov) {
		this.trainCov = trainCov;
	}

	public DoubleMatrix getTestCov() {
		return testCov;
	}

	public void setTestCov(DoubleMatrix testCov) {
		this.testCov = testCov;
	}

	public DoubleMatrix getTestTrainCov() {
		return testTrainCov;
	}

	public void setTestTrainCov(DoubleMatrix testTrainCov) {
		this.testTrainCov = testTrainCov;
	}

	public DoubleMatrix getTrainIn() {
		return trainIn;
	}

	public void setTrainIn(DoubleMatrix trainIn) {
		this.trainIn = trainIn;
	}

	public DoubleMatrix getTestIn() {
		return testIn;
	}

	public void setTestIn(DoubleMatrix testIn) {
		this.testIn = testIn;
	}

	public DoubleMatrix getTrainOut() {
		return trainOut;
	}

	public void setTrainOut(DoubleMatrix trainOut) {
		this.trainOut = trainOut;
	}

	public DoubleMatrix getTestOut() {
		return testOut;
	}

	public void setTestOut(DoubleMatrix testOut) {
		this.testOut = testOut;
	}

}
