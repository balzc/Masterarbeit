package mdp;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import gp.GP;

public class TestMDP {
	public int[][] optPolicy;
	public double[][] expectedRewards;
	public DoubleMatrix predMeanPrice;
	public DoubleMatrix predCovPrice;
	public double[] prices;
	public double deltaPrice;
	public boolean PROFILING = false;
	public double time;
	public double[][][] priceProb;
	public int numSteps;
	public double minp = 0;
	public double maxp = 50;
	public double[] rewards;
	
	public TestMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,
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
		for (int k = 0; k<numSteps-1; k++){
			for (int j = 0; j<prices.length; j++){
				for (int i = 0; i<prices.length; i++){
					priceProb[i][j][k] = priceProb[k][j][i]/normsPrice[k][j];
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
