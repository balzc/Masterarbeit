package mdp;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import gp.GP;

public class HomeHeatingMDP {
	public int[][] optPolicy;
	public double[][] expectedRewards;
	public DoubleMatrix predMeanPrice;
	public DoubleMatrix predCovPrice;
	public DoubleMatrix predMeanInternalTemp;
	public DoubleMatrix predCovInternalTemp;
	public DoubleMatrix predMeanExternalTemp;
	public DoubleMatrix predCovExternalTemp;
	
	public int[] actions;
	public double[] prices;
	public double[] internalTemp;
	public double[] externalTemp;
	public double deltaPrice;
	public double deltaInternalTemp;
	public double deltaExternalTemp;
	public double powerOfHeater;
	public double coefficientOfPerformance;
	public double leakageRate;
	public double massAir;
	public double heatCapacity;
	
	public boolean PROFILING = false;
	public double time;
	public double[][][] priceProb;
	public double[][][][] internalTempProb;
	public double[][][] externalTempProb;
	public int numSteps;
	public double minp = 0;
	public double maxp = 50;
	public double[] rewards;
	
	
	public HomeHeatingMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,
			 int numSteps, double minprice, double maxprice) {
		super();
		this.minp = minprice;
		this.maxp = maxprice;
		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.numSteps = numSteps;
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
	public void computePrices(){
		minp = Math.floor((minp/deltaPrice + 0.5))*deltaPrice;
		int numPrice = (int)(((maxp - minp))/(deltaPrice))+1;
		this.prices = new double[numPrice];
		for(int i = 0; i<numPrice; i++){
			this.prices[i] = minp + i*deltaPrice;
		}
	}
	
	public void computeRewards(){
		rewards = new double[priceProb[0].length];
		for(int i = 0; i < rewards.length; i++){
			rewards[i] = 1;
		}	
	}
	
	public void computeProbabilityTables(){
		if(PROFILING){
			time = System.nanoTime();
		}
		//priceProb[i][j][k] = Pr(price[i]| price[j], timestep = k+1)
		this.priceProb = new double[prices.length][prices.length][numSteps-1];
		//internalTempProb[i][j][k] = Pr(internalTemp[i] | internalTemp[j], externalTemp[k], actions[m])
		this.internalTempProb = new double[internalTemp.length][internalTemp.length][externalTemp.length][2];
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
				}
			}
		}

		//internal temp
		for (int i = 0; i<internalTemp.length;i++){
			for (int j = 0; j<internalTemp.length; j++){
				for (int k = 0; k<externalTemp.length; k++){
					for(int a = 0; a<actions.length; a++){
						internalTempProb[i][j][k][a] = computeInternalTempProb(i, j, k, a);
					}
				}
			}
		}

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
				}
			}
		}
		if(PROFILING){
			System.out.println("computeProbTables - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
		}
	}

	
	public void solveMDP(int timesteps)
	{
		int states = priceProb.length;
		double[][] expectedRewards = new double[states][timesteps];
		int[][] policy = new int[states][timesteps];
		for(int s = 0; s < states; s++){
			expectedRewards[s][timesteps-1] = rewards[s];
		}
		for(int t = timesteps-2; t > -1; t--){
			for(int s = 0; s < states; s++){
				double maxReward = Double.NEGATIVE_INFINITY;
				int bestAction = 0;
//				System.out.println("state: " + s + " at timestep: " + t);

				for(int o = 0; o < priceProb[s].length; o++){
					double eReward = rewards[s];
					for(int i = 0; i < priceProb[s][o].length; i++){
						eReward += priceProb[s][o][i] * expectedRewards[i][t+1];
//						System.out.println(priceProb[s][o][i] + " * " + expectedRewards[i][t+1] + " = "+eReward);

					}
					if(maxReward < eReward){
						maxReward = eReward;
						bestAction = o;
//						System.out.println("updated: " + maxReward);
					}
//					System.out.println("action: " + o + " done");

				}
				expectedRewards[s][t] = maxReward;
				if(priceProb[s].length == 0){
					expectedRewards[s][t] = rewards[s];//expectedRewards[s][t+1]*2; // if no action available keep reward
				}
				policy[s][t] = bestAction;
//				System.out.println("state: " + s + " done");
				

			}
//			System.out.println("timestep: " + t + " done");
//			for(int i = 0; i < states; i++){
//				System.out.print(i + ": " + expectedRewards[i][t]);
//				System.out.println();
//			}

		}
		optPolicy = policy;
		this.expectedRewards = expectedRewards;
		for(int i = 0; i < states; i++){
			System.out.print(i + ": ");

			for(int o = 0; o < timesteps; o++){
				System.out.print(policy[i][o] + " ");
			}
			System.out.println();
		}
		for(int i = 0; i < states; i++){
			System.out.print(i + ": ");

			for(int o = 0; o < timesteps; o++){
				System.out.print(expectedRewards[i][o] + " ");
			}
			System.out.println();
		}
	}
}
