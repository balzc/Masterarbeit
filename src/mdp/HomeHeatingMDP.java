package mdp;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import gp.GP;

public class HomeHeatingMDP {
	private int[][][][] optPolicy;
	private double[][][][] qValues;
	private double[][] expectedRewards;
	private DoubleMatrix predMeanPrice;
	private DoubleMatrix predCovPrice;
	private DoubleMatrix predMeanInternalTemp;
	private DoubleMatrix predCovInternalTemp;
	private DoubleMatrix predMeanExternalTemp;
	private DoubleMatrix predCovExternalTemp;
	
	private int[] actions = {0,1,2};
	private double[] prices;
	private double[] internalTemp;
	private double[] externalTemp;
	private double deltaPrice;
	private double deltaInternalTemp;
	private double deltaExternalTemp;
	private double minInternalTemp;
	private double maxInternalTemp;
	private double sdScale;
	
	private double powerOfHeater;
	private double coefficientOfPerformance;
	private double leakageRate;
	private double massAir;
	private double heatCapacity;
	
	private boolean PROFILING = false;
	private double time;
	private double[][][] priceProb;
	private double[][][][] internalTempProb;
	private double[][][] externalTempProb;
	private int numSteps;

	private double[] rewards;
	
	
	public HomeHeatingMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,
			 int numSteps) {
		super();
		

		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.deltaExternalTemp = deltaPrice;
		this.numSteps = numSteps;
		this.predCovExternalTemp = predCovPrice;
		this.predCovInternalTemp = predCovPrice;
		this.predMeanExternalTemp = predMeanPrice;
		this.predMeanInternalTemp = predMeanPrice;
	}
	
	public void work(){
		computePrices();
		computeExternalTemp();
		computeInternalTemp();
		computeProbabilityTables();
		solveMDP();
	}

	//probability table: priceProb[i][j][k] = Pr(prices[i] | prices[j], timestep=k)
	public double computePriceProb(int i, int j, int k){
		double m = predMeanPrice.get(k) + predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1)*(prices[j] - predMeanPrice.get(k-1));
		double s = Math.sqrt(predCovPrice.get(k,k) - predCovPrice.get(k,k-1)*predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1));

		double y1 = prices[i] - 0.5*deltaPrice;
		double y2 = prices[i] + 0.5*deltaPrice;
		
		return 0.5*(Erf.erf( (y1-m)/(Math.sqrt(2)*s), (y2-m)/(Math.sqrt(2)*s)) );
	}
	public double computeInternalTempProb(int i, int j, int k,int m){
		double Q = actions[m]*powerOfHeater*coefficientOfPerformance - leakageRate*(internalTemp[j] - externalTemp[k]);

		double temp = internalTemp[j] + Q*deltaInternalTemp/(massAir*heatCapacity);
		double prob = Math.abs(temp-internalTemp[i])*2. < deltaExternalTemp? 1 : 0;
		return prob;
	}

	//probability table: externalTempProb[i][j][k] = Pr(exernalTemp[i] | externalTemp[j], timestep= k)
	public double computeExternalTempProb(int i, int j, int k){
		//System.out.println(i+" "+j+" "+" "+k);
		double m = predMeanExternalTemp.get(k) + predCovExternalTemp.get(k,k-1)/predCovExternalTemp.get(k-1,k-1)*(externalTemp[j] - predMeanExternalTemp.get(k-1));
		double s = Math.sqrt(predCovExternalTemp.get(k,k) - predCovExternalTemp.get(k,k-1)*predCovExternalTemp.get(k,k-1)/predCovExternalTemp.get(k-1,k-1));
		//System.out.println("m = "+m+", s = "+s);
		double y1 = externalTemp[i] - 0.5*deltaExternalTemp;
		double y2 = externalTemp[i] + 0.5*deltaExternalTemp;

		return 0.5*(Erf.erf( (y1-m)/(Math.sqrt(2)*s),(y2-m)/(Math.sqrt(2)*s)) );
	}
	public void computeInternalTemp(){
		int numTemp = (int)(((maxInternalTemp - minInternalTemp))/(deltaExternalTemp))+1;
		this.internalTemp = new double[numTemp];
		for (int i = 0; i<numTemp; i++){
			internalTemp[i] = minInternalTemp + i*deltaExternalTemp;
		}
	}

	public void computeExternalTemp(){
		double maxExternalTemp = findMaximumExternalTemp();
		double minExternalTemp = findMinimumExternalTemp();

		minExternalTemp = Math.floor((minExternalTemp/deltaExternalTemp + 0.5))*deltaExternalTemp;

		int numTemp = (int)(((maxExternalTemp - minExternalTemp))/(deltaExternalTemp))+1;
		this.externalTemp = new double[numTemp];

		for (int i = 0; i<numTemp; i++){
			externalTemp[i] = minExternalTemp + i*deltaExternalTemp;
		}

	}

	public double findMaximumExternalTemp(){
		double maxTemp = Double.MIN_VALUE;
		double t;
		for (int i = 0; i<predMeanExternalTemp.length; i++){
			t = predMeanExternalTemp.get(i) + sdScale*Math.sqrt(predCovExternalTemp.get(i,i));
			if(t > maxTemp){
				maxTemp = t;
			}
		}
		return maxTemp;
	}

	public double findMinimumExternalTemp(){
		double minTemp = Double.MAX_VALUE;
		double t;
		for (int i = 0; i<predMeanExternalTemp.length; i++){
			t = predMeanExternalTemp.get(i) - sdScale*Math.sqrt(predCovExternalTemp.get(i,i));
			if(t < minTemp){
				minTemp = t;
			}
		}
		return minTemp;
	}

	public void computePrices(){
		double minPrice = findMinimumPrice();
		double maxPrice = findMaximumPrice();

		minPrice = Math.floor((minPrice/deltaPrice + 0.5))*deltaPrice;

		int numPrice = (int)(((maxPrice - minPrice))/(deltaPrice))+1;
		this.prices = new double[numPrice];
		for(int i = 0; i<numPrice; i++){
			this.prices[i] = minPrice + i*deltaPrice;
		}
	}

	public double findMaximumPrice(){
		double maxPrice = Double.MIN_VALUE;
		double t;
		for(int i = 0; i<predMeanPrice.length; i++){
			t = predMeanPrice.get(i) + sdScale*Math.sqrt(predCovPrice.get(i,i));
			if(t>maxPrice){
				maxPrice = t;
			}
		}
		return maxPrice;
	}

	public double findMinimumPrice(){
		double minPrice = Double.MAX_VALUE;
		double t;
		for(int i = 0; i<predMeanPrice.length; i++){
			t = predMeanPrice.get(i) - sdScale*Math.sqrt(predCovPrice.get(i,i));
			if(t < minPrice){
				minPrice = t;
			}
		}
		return minPrice;
	}
	
	public void computeRewards(){
		rewards = new double[priceProb[0].length];
		for(int i = 0; i < rewards.length; i++){
			rewards[i] = 1;
		}	
	}
	
	public double rewards(double internalTemp, int action, double price){return Math.random();}
	
	
	public void computeProbabilityTables(){
		if(PROFILING){
			time = System.nanoTime();
		}
		//priceProb[i][j][k] = Pr(price[i]| price[j], timestep = k+1)
		this.priceProb = new double[prices.length][prices.length][numSteps-1];
		//internalTempProb[i][j][k] = Pr(internalTemp[i] | internalTemp[j], externalTemp[k], actions[m])
		this.internalTempProb = new double[internalTemp.length][internalTemp.length][externalTemp.length][actions.length];
		//externalTempProb[i][j][k] = Pr(exernalTemp[i] | externalTemp[j], timestep= k+1)
		this.externalTempProb = new double[externalTemp.length][externalTemp.length][numSteps-1];

		//prices
		double[][] normsPrice = new double[numSteps-1][prices.length];
		for (int k = 0; k<numSteps-1; k++){
			for (int j = 0; j<prices.length; j++){
				for (int i = 0; i<prices.length; i++){
					double t = computePriceProb(i, j, k+1);
					priceProb[i][j][k] = t;
					normsPrice[k][j] += t;
				}
			}
		}
		//normalize externalTempProb
		for (int k = 0; k<numSteps-1; k++){
			for(int j = 0; j<prices.length; j++){
				for (int i = 0; i<prices.length; i++){
					priceProb[i][j][k] /= normsPrice[k][j];
					System.out.print(priceProb[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();

		}
		System.out.println();
		//internal temp
		for (int i = 0; i<internalTemp.length;i++){
			for (int j = 0; j<internalTemp.length; j++){
				for (int k = 0; k<externalTemp.length; k++){
					for(int a = 0; a<actions.length; a++){
						internalTempProb[i][j][k][a] = computeInternalTempProb(i, j, k, a);
						System.out.print(internalTempProb[i][j][k][a] + " ");

					}
					System.out.println();

				}
				System.out.println();

			}		
			System.out.println();

		}
		System.out.println();

		//external temp
		double[][] normsTemp = new double[numSteps-1][externalTemp.length];
		for (int k = 0; k<numSteps-1; k++){
			for (int j = 0; j<externalTemp.length; j++){
				for (int i = 0; i<externalTemp.length; i++){
					double t = computeExternalTempProb(i, j, k+1);
					externalTempProb[i][j][k] = t;
					normsTemp[k][j] += t;
				}
			}
		}
		//normalize externalTempProb
		for (int k = 0; k<numSteps-1; k++){
			for(int j = 0; j<externalTemp.length; j++){
				for (int i = 0; i<externalTemp.length; i++){
					externalTempProb[i][j][k] /= normsTemp[k][j];
					System.out.print(externalTempProb[i][j][k] + " ");

				}
				System.out.println();

			}
			System.out.println();

		}
		System.out.println();

		if(PROFILING){
			System.out.println("computeProbTables - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
		}
	}


	public void solveMDP()
	{	
		qValues = new double[numSteps][prices.length][internalTemp.length][externalTemp.length];
		optPolicy = new int[numSteps][prices.length][internalTemp.length][externalTemp.length];
		for(int p = 0; p < prices.length; p++){
			for(int it = 0; it < internalTemp.length; it++){
				for(int et = 0; et < externalTemp.length; et++){
					double maxReward = Double.NEGATIVE_INFINITY;
					for(int a = 0; a < actions.length; a++){
						double temp = rewards(it,a,p);
						if(temp > maxReward){
							qValues[numSteps-1][p][it][et] = temp;
							maxReward = temp;
						}
					}

				}
			}
		}

		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < internalTemp.length; it++){
					for(int et = 0; et < externalTemp.length; et++){
						double currentMax = Double.NEGATIVE_INFINITY;
						int currentBestAction = 0;
						for(int a = 0; a < actions.length; a++){
							double qval = rewards(it,a,p);
							for(int pn = 0; pn < prices.length; pn++){
								for(int itn = 0; itn < internalTemp.length; itn++){
									for(int etn = 0; etn < externalTemp.length; etn++){
										qval += internalTempProb[itn][it][et][a]*externalTempProb[etn][et][t]*priceProb[p][pn][t]*qValues[t+1][pn][itn][etn];
									}
								}
							}

							if(qval > currentMax){
								currentMax = qval;
								currentBestAction = a;
								qValues[t][p][it][et] = qval;
								optPolicy[t][p][it][et] = currentBestAction;
							}
						}
					}
				}
			}
		}
	}
	
	public void printOptPolicy(){
		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < internalTemp.length; it++){
					for(int et = 0; et < externalTemp.length; et++){
						System.out.print(optPolicy[t][p][it][et] + " ");
					}
				}
			}
			System.out.println();
		}
	}
	public void printQvals(){
		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < internalTemp.length; it++){
					for(int et = 0; et < externalTemp.length; et++){
						System.out.print(qValues[t][p][it][et] + " ");
					}
				}
			}
			System.out.println();
		}
	}
}
