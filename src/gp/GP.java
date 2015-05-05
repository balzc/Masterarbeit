package gp;

import cov.CovarianceFunction;
import cov.SquaredExponential;

import org.jblas.Decompose;
import org.jblas.Solve;
import org.jblas.DoubleMatrix;
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
	
	private int numTest;
	private int numTrain;
	
	public GP(DoubleMatrix trainInput,DoubleMatrix trainOutput, DoubleMatrix testInput, CovarianceFunction cf, double noisel){
		noiselevel = noisel;
		trainIn = trainInput;
		testIn = testInput;
		trainOut = trainOutput;
		covf = cf;
		numTest = testIn.columns;
		numTrain =trainIn.columns;
		DoubleMatrix parameters = DoubleMatrix.ones(2);

		DoubleMatrix identity = DoubleMatrix.eye(trainIn.columns);
		trainCov = computeCovMatrix(trainIn, trainIn, parameters);
		testTrainCov = computeCovMatrix(trainIn, testIn, parameters);
		testCov = computeCovMatrix(testIn, testIn, parameters);
		
		DoubleMatrix temp = trainCov.add(identity.mul(noiselevel));
		temp.print();
		l = Decompose.cholesky(temp);
		
		DoubleMatrix param = minimize(parameters, -100, trainIn, trainOut);
		System.out.println("Parameters:");
		param.print();
		
		trainCov.print();
		
		
		l.print();
		DoubleMatrix beta = Solve.pinv(l).mmul(Solve.pinv(l).mmul(trainOut.transpose()));
		alpha = Solve.solve(l.transpose(), Solve.solve(l, trainOut.transpose()));
		alpha.print();
		beta.print();
		predMean = testCov.transpose().mmul(alpha);
		testCov.print();
		predVar = Solve.solve(l, testCov);
		predVar.print();
		
		
		System.out.println(logLikelihood);
	}
	public DoubleMatrix calculateAlpha(DoubleMatrix trainInput,  DoubleMatrix trainOutput, DoubleMatrix parameters, double noiselvl){
		DoubleMatrix cova = computeCovMatrix(trainInput, trainInput,parameters);
		DoubleMatrix identity = DoubleMatrix.eye(trainInput.columns);
		DoubleMatrix temp = cova.add(identity.mul(noiselvl));
		DoubleMatrix el = Decompose.cholesky(temp);
		DoubleMatrix beta = Solve.pinv(el).mmul(Solve.pinv(el).mmul(trainOutput.transpose()));
		return Solve.solve(el.transpose(), Solve.solve(el, trainOutput.transpose()));
	}
	public double calcuateMean(){return 0;}
	public double calculateVariance(){return 0;}
	public double calculateLogLikelihood(){return 0;}

	public double negativeLogLikelihood(DoubleMatrix parameters, DoubleMatrix in, DoubleMatrix out, DoubleMatrix df0){
		double loglike = out.transpose().mul(calculateAlpha(in, out, parameters, noiselevel)).get(0,0)*(-0.5);
		for(int i = 0; i < l.rows; i++){
			loglike -= Math.log(l.get(i, i));
		}
		loglike -= (numTrain/2)*Math.log(2*Math.PI);
		return -loglike;
	}
	
	// compute the covariance matrix of two vector inputs TODO: double computations unnecessary when k = kstar
	public DoubleMatrix computeCovMatrix(DoubleMatrix k, DoubleMatrix kstar, DoubleMatrix parameters){
		DoubleMatrix result = new DoubleMatrix(k.columns,kstar.columns);
		for(int i = 0; i<1; i++){
			for(int j = 0; j<1; j++){
				result.put(i, j, covf.computeCovariance(k.getColumn(i), kstar.getColumn(j), parameters));
			}
		}
		return result;
	}
	
	
    private final static double INT = 0.1;                // don't reevaluate within 0.1 of the limit of the current bracket

    private final static double EXT = 3.0;                // extrapolate maximum 3 times the current step-size

    private final static int MAX = 20;                    // max 20 function evaluations per line search

    private final static double RATIO = 10;               // maximum allowed slope ratio

    private final static double SIG = 0.1, RHO = SIG/2;   // SIG and RHO are the constants controlling the Wolfe-
    // Powell conditions. SIG is the maximum allowed absolute ratio between
    // previous and new slopes (derivatives in the search direction), thus setting
    // SIG to low (positive) values forces higher precision in the line-searches.
    // RHO is the minimum allowed fraction of the expected (from the slope at the
    // initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    // Tuning of SIG (depending on the nature of the function to be optimized) may
    // speed up the minimization; it is probably not worth playing much with RHO.

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
    private DoubleMatrix minimize(DoubleMatrix params, int length, DoubleMatrix in, DoubleMatrix out){

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
        //f0 = f.evaluate(params,cf, in, out, df0);

        fX = new DoubleMatrix(new double[]{f0});

        i = (length < 0) ? i+1 : i;

        DoubleMatrix s = df0.mul(-1);

        // initial search direction (steepest) and slope
        d0 = s.mul(-1).transpose().mul(s).get(0,0);
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

                    DoubleMatrix m1 = params.add(s.mul(x3));
                    //f3 = f.evaluate(m1,cf, in, out, df3);
                    f3 = negativeLogLikelihood(m1, in, out,df3);

                    if (Double.isNaN(f3) || Double.isInfinite(f3) || hasInvalidNumbers(df3.toArray())){
                        x3 = (x2+x3)/2;     // catch any error which occured in f
                    }else{
                        success = 1;
                    }

                }

                if (f3 < F0){                   // keep best values
                    X0 = s.mul(x3).add(params);
                    F0 = f3;
                    dF0 = df3;
                }

                d3 = df3.transpose().mul(s).get(0,0);  // new slope

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

                DoubleMatrix m1 = s.mul(x3).add(params);
                //f3 = f.evaluate(m1,cf, in, out, df3);
                f3 = negativeLogLikelihood(m1, in, out,df3);

                if (f3 < F0){
                    X0 = m1.dup();
                    F0 = f3;
                    dF0 = df3.dup();                            // keep best values
                }

                M = M - 1;  i = (length < 0) ? i+1 : i;          // count iterations?!

                d3 = df3.transpose().mul(s).get(0,0); // new slope

            }                                                    // end interpolation

            if (Math.abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0){     // if line search succeeded
                params = s.mul(x3).add(params);
                f0 = f3;

                double[] elem = fX.toArray();
                double[] newfX = new double[elem.length + 1];

                System.arraycopy(elem, 0, newfX, 0, elem.length);
                newfX[elem.length-1] = f0;
                fX = new DoubleMatrix(newfX);                 // update variables


                System.out.println("Function evaluation "+i+" Value "+f0);

                
                double tmp1 = df3.transpose().mul(df3).sub(df0.transpose().mul(df3)).get(0,0);
                double tmp2 = df0.transpose().mul(df0).get(0,0);

                s =  s.mul(tmp1/tmp2).sub(df3);

                df0 = df3;                          // swap derivatives
                d3 = d0;
                d0 = df0.transpose().mul(s).get(0,0);

                if (d0 > 0){                        // new slope must be negative
                    s = df0.mul(-1);              // otherwise use steepest direction
                    d0 = s.mul(-1).transpose().mul(s).get(0,0);
                }

                x3 = x3 * Math.min(RATIO, d3/(d0-Double.MIN_VALUE));    // slope ratio but max RATIO
                ls_failed = 0;                                          // this line search did not fail

            }else{

                params = X0; f0 = F0; df0 = dF0;                     // restore best point so far

                if (ls_failed == 1 || i > Math.abs(length)){    // line search failed twice in a row
                    break;                                      // or we ran out of time, so we give up
                }

                s = df0.mul(-1); d0 = s.mul(-1).transpose().mul(s).get(0,0);      // try steepest
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
}
