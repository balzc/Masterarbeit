package mdp;

import java.util.Arrays;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import main.Main;
import gp.GP;

public class EVMDP {
	private int[][][] optPolicy;
	private double[][][] qValues;
	private DoubleMatrix predMeanPrice;
	private DoubleMatrix predCovPrice;

	private int[] actions = {0,1};
	private double[] prices;
	private int[] loads;
	private double deltaPrice;
	private int qMax;
	private int qMin;
	private int qRequired;
	private double qSlope;
	private double qOffset;
	private double tSlope;
	private double tOffset;
	private double vTBase;
	private double vQBase;
	private int tStart;
	private int tCrit;
	private double sdScale;



	private boolean PROFILING = false;
	private double time;
	private double[][][] priceProb;
	private double[][][] loadProb;
	private int numSteps;

	double tol = 0.0001;


	public EVMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,	int numSteps) {


		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = 40;//deltaPrice;
		this.numSteps = numSteps;
		this.sdScale = 5;
		this.qMax = 25;
		this.qRequired = 10;
		this.qSlope = 2;
		this.tOffset = 0;
		this.tStart = 80;
		this.tCrit = 82;
		this.vTBase = 10;
		this.tSlope = -(vTBase)/(tCrit-tStart);
		this.vQBase = 10;

	}

	public void work(){
		//		System.out.println(rewards(18, 0, 100) + " " + rewards(16, 0, 100) + " " + rewards(17, 1, 100));
		computePrices();


		computeLoad();
		computeProbabilityTables();

		solveMDP();
		//		printOptPolicy();
	}

	//probability table: priceProb[i][j][k] = Pr(prices[i] | prices[j], timestep=k)
	public double computePriceProb(int i, int j, int k){
//		double m = predMeanPrice.get(k) + predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1)*(prices[j] - predMeanPrice.get(k-1));
//		double s = Math.sqrt(predCovPrice.get(k,k) - predCovPrice.get(k,k-1)*predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1));
//
//		double y1 = prices[i] - 0.5*deltaPrice;
//		double y2 = prices[i] + 0.5*deltaPrice;
//		return 0.5*(Erf.erf( (y1-m)/(Math.sqrt(2)*s), (y2-m)/(Math.sqrt(2)*s)) );
				if(prices[i] == 0 && (k < 24 ) ){
					return 1;
				}else if(prices[i] == 0 && (k < 48 && k > 23) ){
					return 1;
				} else if(prices[i] == 40 && (k < 96 && k >= 48) ){
					return 1;
				} else {
					return 0;
				}

	}
	public double computeLoadProb(int i, int j, int m){

		if(i-j == m){
			return 1;
		}
		else if (i == qMax && j + m > qMax){
			return 1;
		}
		else{
			return 0;
		}
		
	}


	public void computeLoad(){
		this.loads = new int[qMax+1];
		for (int i = 0; i<qMax+1; i++){
			loads[i] = i;
		}
	}




	public void computePrices(){
		double minPrice = 0;//findMinimumPrice();
		double maxPrice = 40;//findMaximumPrice();

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


	public double rewards(int load, int action, double price, int time){
		double cost = price * action;
		if(load < qRequired){
			return 0 - cost;
		}
		else {
			if(time >= tStart && time <= tCrit){
				return (qOffset + qSlope*time)*(tOffset + tSlope*time) - cost;
			}
			else if(time < tStart){
				return (qOffset + qSlope*time)*vTBase - cost;
			}
			else {
				return 0 - cost;
			}
		}
	}






	public int priceToState(double price){
		int index = -1;
		double p = roundToNextPrice(price, deltaPrice);
		for(int i = 0; i<prices.length; i++){
			if (Math.abs(prices[i]-p)<tol) {
				index = i;
				break;
			}
		}
		return index;
	}






	public double roundToNextPrice(double price, double delta){
		return Math.floor((price/delta + 0.5))*delta;
	}


	public int updateLoad(int initialLoad, int action){
		if(initialLoad + action > qMax){
			return qMax;
		}
		else{
			return initialLoad + action;
		}

	}

	public void computeProbabilityTables(){
		if(PROFILING){
			time = System.nanoTime();
		}
		//priceProb[i][j][k] = Pr(price[i]| price[j], timestep = k+1)
		this.priceProb = new double[prices.length][prices.length][numSteps-1];
		//internalTempProb[i][j][k] = Pr(internalTemp[i] | internalTemp[j], externalTemp[k], actions[m])
		this.loadProb = new double[loads.length][loads.length][actions.length];
		//externalTempProb[i][j][k] = Pr(exernalTemp[i] | externalTemp[j], timestep= k+1)

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
			//			System.out.print("k= " +k + " ");

			for(int j = 0; j<prices.length; j++){
				//				System.out.print("j= "+j + " ");

				for (int i = 0; i<prices.length; i++){
					priceProb[i][j][k] /= normsPrice[k][j];
					//					System.out.println("i=" + i + " ");

					//					System.out.print(priceProb[i][j][k] + " ");
				}
				//				System.out.println();
			}
			//			System.out.println();

		}
		//		System.out.println();
		//internal temp
		for (int i = 0; i<loads.length;i++){
			for (int j = 0; j<loads.length; j++){
				for(int a = 0; a<actions.length; a++){
					loadProb[i][j][a] = computeLoadProb(i, j, a);
					//						System.out.print(internalTempProb[i][j][k][a] + " ");

				}
				//					System.out.println();


				//				System.out.println();

			}		
			//			System.out.println();

		}
		//		System.out.println();

		//external temp

		if(PROFILING){
			System.out.println("computeProbTables - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
		}
	}


	public void solveMDP()
	{	
		qValues = new double[numSteps][prices.length][loads.length];

		optPolicy = new int[numSteps][prices.length][loads.length];
		for(int p = 0; p < prices.length; p++){
			for(int it = 0; it < loads.length; it++){


				double temp = rewards(loads[it],0,prices[p],numSteps-1);
				qValues[numSteps-1][p][it] = temp;



			}
		}


		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < loads.length; it++){
					double currentMax = Double.NEGATIVE_INFINITY;
					int currentBestAction = 0;
					int counter[] = {0,0};

					for(int a = 0; a < actions.length; a++){
						double qval = rewards(loads[it],a,prices[p],t);
						//							System.out.println("action: " +a);
						double sumextt = 0;
						double sumintt = 0;
						double sump = 0;
						double sumtot = 0;
						for(int pn = 0; pn < prices.length; pn++){
							for(int itn = 0; itn < loads.length; itn++){
								qval += loadProb[itn][it][a]*priceProb[pn][p][t]*qValues[t+1][pn][itn];


								//										if(Math.abs(loadProb[itn][it][et][a]*externalTempProb[etn][et][t]*priceProb[pn][p][t]-1)<Math.pow(10, -10)){
								//											counter[a]++;
								//										}
								//										sumextt += externalTempProb[etn][et][t];
								//										sumintt += loadProb[itn][it][et][a];
								//										sump += priceProb[pn][p][t];
								//										sumtot += loadProb[itn][it][et][a]*externalTempProb[etn][et][t]*priceProb[pn][p][t];


							}
						}


						//							if(qval- rewards(internalTemp[it],a,prices[p])< 10){
						//								System.out.println("sumextt: " +sumextt + " sump:  " + sump + " sumintt " + sumintt + " qval: " + qval + " sumtot " +sumtot);
						//							}
						//							
						if(qval > currentMax){
							currentMax = qval;
							currentBestAction = a;
							//								if(currentBestAction == 1){
							//									System.out.println(qValues[t][p][it][et] + " se vals " + qval);
							//									System.out.println("Rewards:" +rewards(internalTemp[it],0,prices[p]) +" " +rewards(internalTemp[it],1,prices[p]));
							//									System.out.println("counter0: " + counter[0] + " counter1:  " + counter[1] );
							//
							//
							//								}
							qValues[t][p][it] = qval;
							optPolicy[t][p][it] = currentBestAction;



						} 
					}


				}
			}
		}

		//		for(int j = 0; j<prices.length;j++){
		//			for(int i = 0; i<prices.length;i++){
		//				System.out.print("newold "+i+ " " + j + " ");
		//				for(int c = 0; c < numSteps-1;c++ ){
		//					System.out.print(externalTempProb[j][i][c]+ " ");
		//				}
		//			}
		//			System.out.println();
		//		}

		printLoads();
		printPrices();
		System.out.println("test " );
		for(int k = numSteps-1; k >0; k--){
			for(int j = 0; j < prices.length; j++){
				System.out.print(k + " [ ");
				for(int i = 12; i<loads.length;i++){
					System.out.print(i+ ":" + qValues[k][j][i]+ " ");

				}
				System.out.println(" ]");
			}
		}
		System.out.println("testend");

	}

	public void printOptPolicy(){
		double sum = 0;
		for(int t = 0; t <numSteps; t++){
			System.out.print(t + "  ");

			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < loads.length; it++){
					if(p == 1 && optPolicy[t][p][it] == 1){
						System.out.print(prices[p] + " " + loads[it] + " "  + optPolicy[t][p][it] + "; ");}
					sum += optPolicy[t][p][it];

				}
				System.out.println();

			}
			System.out.println();
		}
		System.out.println("Sum: " + sum);
	}
	public void printQvals(){
		for(int t = 0; t <numSteps; t++){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < loads.length; it++){
					System.out.print(prices[p] + " " + loads[it] + " " +  qValues[t][p][it]+ "; ");

				}
			}
			System.out.println();
		}
	}
	public void printPrices(){
		for(int i = 0; i<prices.length;i++){
			System.out.print(prices[i] + " ");
		}
		System.out.println();

	}


	public void printLoads(){
		for(int i = 0; i<loads.length;i++){
			System.out.print(loads[i] + " ");
		}
		System.out.println();

	}






	// getters and setters

	public int[][][] getOptPolicy() {
		return optPolicy;
	}

	public void setOptPolicy(int[][][] optPolicy) {
		this.optPolicy = optPolicy;
	}

	public double[][][] getqValues() {
		return qValues;
	}

	public void setqValues(double[][][] qValues) {
		this.qValues = qValues;
	}






}
