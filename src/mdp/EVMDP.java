package mdp;

import java.util.Arrays;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import main.Main;
import gp.GP;

public class EVMDP {
	private int[][][][] optPolicy;
	private double[][][][] qValues;
	private DoubleMatrix predMeanPrice;
	private DoubleMatrix predCovPrice;

	private int[] actions = {0,1};
	private double[] prices;
	private double[] loads;
	private double deltaPrice;
	private double qMax;
	private double qInitial;
	private double qRequired;
	private double qSlope;
	private double vMin;
	private double tStart;
	private double tCrit;
	private double sdScale;
	
	private double tMean;
	private double tSD;
	
	private boolean PROFILING = false;
	private double time;
	private double[][][] priceProb;
	private double[][][] loadProb;
	private double[][][] endStateProb;

	private int numSteps;

	private double kwhPerUnit;
	
	double tol = 0.0001;


	public EVMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,	int numSteps) {

		double value = 14.;
		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.numSteps = numSteps;
		this.sdScale = deltaPrice;
		this.qMax = 5;//3
		this.qRequired = 2;//2
		this.qSlope = 161;//15
		this.tStart = 5;//6
		this.tCrit = 7;//9
		this.vMin = 321;//14*qRequired
		this.tMean = 7;//5
		this.tSD = 0.5;//0.01
		this.qInitial = 0;
		// deltat = (tcrit-tplug)/numsteps
	}

	public EVMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,	int numSteps, double qMax, double qRequired, double qSlope, double tStart, double tCrit, double vMin, double tmean, double tsd, double qinitial, double wattPerUnitInput) {


		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.numSteps = numSteps;
		this.sdScale = 6*deltaPrice;
		this.qMax = qMax;
		this.qRequired = qRequired;
		this.qSlope = qSlope;
		this.tStart = tStart;
		this.tCrit = tCrit;
		this.vMin = vMin;
		this.tMean = tmean;
		this.tSD = tsd;
		this.qInitial = qinitial;
		this.kwhPerUnit = wattPerUnitInput;
		// deltat = (tcrit-tplug)/numsteps
	}
	public EVMDP( double qSlope, double vMin, double wattPerUnitInput, double[][][] priceprobs, double[][][] loadprobs, double[][][] endstateprobs, double[] prices ,double[] loads, int numsteps,double tstart, double tcrit, double tmean, double qrequired, double qmax) {


	
		this.sdScale = 6*deltaPrice;
		this.priceProb = priceprobs;
		this.loadProb = loadprobs;
		this.endStateProb = endstateprobs;
		this.prices = prices;
		this.loads = loads;
		this.qSlope = qSlope;
		this.numSteps = numsteps;
		this.vMin = vMin;
		this.tMean = tmean;
		this.tCrit = tcrit;
		this.tStart = tstart;
		this.kwhPerUnit = wattPerUnitInput;
		this.qRequired = qrequired;
		this.qMax = qmax;
		// deltat = (tcrit-tplug)/numsteps
	}
	
	public void setup(){
		
		computePrices();


		computeLoad();
		computeProbabilityTables();
		solveMDP();
//		printQvals();
//		System.out.println("rewardtest " + rewards(879, 1, 100, 8, 0));
//		printLoads();
//
//		for(int i = 0; i < numSteps-1; i++){
//			System.out.println(i + ": " + endStateProb[1][0][i] );
//		}
//		printPrices();
//		printOptPolicy();
	}
	public void fastSetup(){
		solveMDP();
	}
	
	
	public double computeEndStateProb(int i, int j, int t){
		if(i == j && i == 0){
			return 1;
		} else if (i == j && i == 1){
			return 0;
		} else if(i == 1 && j == 0){
//			if(Math.abs(t - tMean) < 0.001){
//				return 0.5;
//			}
//			else if(Math.abs(t - tMean)-1 < 0.001){
//				return 0.25;
//			}
//			else {
//				return 0;
//			}
			return 0.5*(1 + Erf.erf(((t+0.5)-tMean)/(tSD * Math.sqrt(2))))-0.5*(1 + Erf.erf(((t-0.5)-tMean)/(tSD * Math.sqrt(2))));
		} else {
			return 0;
		}
	}
	
	
	//probability table: priceProb[i][j][k] = Pr(prices[i] | prices[j], timestep=k)
	public double computePriceProb(int i, int j, int k){
		double m = predMeanPrice.get(k) + predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1)*(prices[j] - predMeanPrice.get(k-1));
		double s = Math.sqrt(predCovPrice.get(k,k) - predCovPrice.get(k,k-1)*predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1));

		double y1 = prices[i] - 0.5*deltaPrice;
		double y2 = prices[i] + 0.5*deltaPrice;
		return 0.5*(Erf.erf( (y1-m)/(Math.sqrt(2)*s), (y2-m)/(Math.sqrt(2)*s)) );
//				if(prices[i] == 20 && (k < 3 ) ){
//					return 1;
//				}else if(prices[i] == 30 && (k < 6 && k > 2) ){
//					return 1;
//				} else if(prices[i] == 10 && (k < 9 && k > 5) ){
//					return 1;
//				} else if(prices[i] == 10 && (k < 12 && k > 8) ){
//					return 1;
//				} else {
//					return 0;
//				}

	}
	public double computeLoadProb(int i, int j, int m){

		if(i-j == m){
			return 1;
		}
		else if (i == qMax && j + m > qMax ){
			return 1;
		}
		else{
			return 0;
		}

	}

	// Tested, Works as intended
	public void computeLoad(){
		int qRange = (int)(qMax-qInitial/kwhPerUnit+1);
		if(qInitial + qRange-1 * kwhPerUnit < qMax * kwhPerUnit){
			this.loads = new double[qRange+1];
			loads[qRange] = qMax*kwhPerUnit;
		} else {
			this.loads = new double[qRange];
		}
		for (int i = 0; i<qRange; i++){
			loads[i] = qInitial + i * kwhPerUnit;
		}
	}



	// Tested, Works as intended
	public void computePrices(){
		double minPrice = 0;//findMinimumPrice();
		double maxPrice = 180;//findMaximumPrice();

		minPrice = Math.floor((minPrice/deltaPrice + 0.5))*deltaPrice;

		int numPrice = (int)(((maxPrice - minPrice))/(deltaPrice))+1;
		this.prices = new double[numPrice];
		for(int i = 0; i<numPrice; i++){
			this.prices[i] = minPrice + i*deltaPrice;
		}
	}
	// Tested, Works as intended
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
	// Tested, Works as intended
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


	public double rewards(double q, int action, double price, int t, int endState){
		double loadToMax = Math.min((qMax*kwhPerUnit-q),kwhPerUnit);
		double cost = price * action * loadToMax;
//		if(value(q+action * wattPerUnit,t+1) >= value(q,t)){
		if(endState == 1){
			return value(q,t);
		}
		else {
			return -cost;
		}
//		}
//		else{
//			return 0;
//		}

	}

	public double value(double qin, int t){
		double q = Math.min(qin, qMax*kwhPerUnit);
		double requiredLoad = qRequired*kwhPerUnit;
		double qflex = Math.max((q-requiredLoad), 0);
		if(q >= qRequired*kwhPerUnit){
			if(t < tStart){
				return (vMin + qSlope*qflex);
			}
			else if(t < tCrit){
				return (vMin + qSlope*qflex)*((tCrit-t)/(tCrit-tStart));
			}
			else{
				return 0;
			}
				
		} else {
			return 0;
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

	public int loadToState(double load){
		int index = -1;

		for(int i = 0; i<loads.length; i++){
			if (Math.abs(loads[i]-load)<tol) {
				index = i;
				break;
			}
		}

		return index;
	}




	public double roundToNextPrice(double price, double delta){
		return Math.floor((price/delta + 0.5))*delta;
	}


	public double updateLoad(double initialLoad, int action){
		if(initialLoad + action*kwhPerUnit > qMax*kwhPerUnit){
			return qMax*kwhPerUnit;
		}
		else{
			return initialLoad + action*kwhPerUnit;
		}

	}

	public void computeProbabilityTables(){
		if(PROFILING){
			time = System.nanoTime();
		}
		//priceProb[i][j][k] = Pr(price[i]| price[j], timestep = k+1)
		this.priceProb = new double[prices.length][prices.length][numSteps-1];
		this.endStateProb = new double[2][2][numSteps];
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

					//										System.out.print(priceProb[i][j][k] + " ");
				}
				//								System.out.println();
			}
			//						System.out.println();

		}
		//		System.out.println();
		//internal temp
		for (int i = 0; i<loads.length;i++){
			for (int j = 0; j<loads.length; j++){
				for(int a = 0; a<actions.length; a++){
					loadProb[i][j][a] = computeLoadProb(i, j, a);
//																System.out.print(loadProb[i][j][a] + " ");

				}
//													System.out.println();



			}		
//									System.out.println();

		}
		//		System.out.println();
		for(int i = 0; i < 2; i++){
			for(int j = 0; j < 2; j++){
				for(int t = 0; t < numSteps; t++){
					endStateProb[i][j][t] = computeEndStateProb(i, j, t);
				}
			}
		}
		//external temp

		if(PROFILING){
			System.out.println("computeProbTables - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
		}
	}


	public void solveMDP()
	{	
		qValues = new double[numSteps][prices.length][loads.length][2];

		optPolicy = new int[numSteps][prices.length][loads.length][2];
		for(int p = 0; p < prices.length; p++){
			for(int q = 0; q < loads.length; q++){


				double temp = rewards(loads[q],0,prices[p],numSteps-1,0);
				qValues[numSteps-1][p][q][0] = temp;



			}
		}


		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int q = 0; q < loads.length; q++){
					for(int e = 0; e < 2; e++){
						double currentMax = Double.NEGATIVE_INFINITY;
						int currentBestAction = 0;
						int counter[] = {0,0};

						for(int a = 0; a < actions.length; a++){
							double qval = rewards(loads[q],a,prices[p],t,e);
							//							System.out.println("action: " +a);
							double sumextt = 0;
							double sumintt = 0;
							double sump = 0;
							double sumtot = 0;
							if(e != 1){
								for(int pn = 0; pn < prices.length; pn++){
									for(int qn = 0; qn < loads.length; qn++){
										for(int en = 0; en < 2; en++){
											//										if(t == 2 && loadProb[qn][q][a]*priceProb[pn][p][t]*qValues[t+1][pn][qn][en]*endStateProb[en][e][t+1] > 0){
											//											System.out.println("a test " + qn + " " + q + " p " + pn + " " + p + " " + loadProb[qn][q][a]*priceProb[pn][p][t]*qValues[t+1][pn][qn][en]*endStateProb[en][e][t+1]);
											//										}
											qval += loadProb[qn][q][a]*priceProb[pn][p][t]*qValues[t+1][pn][qn][en]*endStateProb[en][e][t+1];
										}
									}
								}	
							}
							if(qval > currentMax){
								currentMax = qval;
								currentBestAction = a;
								qValues[t][p][q][e] = qval;
								optPolicy[t][p][q][e] = currentBestAction;



							} 
						}
					}


				}
			}
		}


	}

	public void printOptPolicy(){
		double sum = 0;
		for(int t = 0; t <numSteps; t++){
			System.out.println(t + "  ");

			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < loads.length; it++){
					System.out.print(prices[p] + " " + loads[it] + " "  + optPolicy[t][p][it][0] + "; ");
//					sum += optPolicy[t][p][it][0];

				}
				System.out.println();

			}
			System.out.println();

		}
//		System.out.println("Sum: " + sum);
	}

public void printQvals(){
	for(int t = 0; t <numSteps; t++){
		for(int p = 0; p < prices.length; p++){
			for(int it = 0; it < loads.length; it++){
//				for(int e = 0; e < 2; e++){
					if(qValues[t][p][it][0] != 0){
						System.out.print( t + " "+ prices[p] + " " + loads[it] + " " +  qValues[t][p][it][0]+ "; ");
					}
//				}
			}
			System.out.println();
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

public int[][][][] getOptPolicy() {
	return optPolicy;
}

public void setOptPolicy(int[][][][] optPolicy) {
	this.optPolicy = optPolicy;
}

public double[][][][] getqValues() {
	return qValues;
}

public void setqValues(double[][][][] qValues) {
	this.qValues = qValues;
}

public double[] getPrices() {
	return prices;
}

public void setPrices(double[] prices) {
	this.prices = prices;
}

public double[] getLoads() {
	return loads;
}

public void setLoads(double[] loads) {
	this.loads = loads;
}

public double[][][] getPriceProb() {
	return priceProb;
}

public void setPriceProb(double[][][] priceProb) {
	this.priceProb = priceProb;
}

public double[][][] getLoadProb() {
	return loadProb;
}

public void setLoadProb(double[][][] loadProb) {
	this.loadProb = loadProb;
}

public double[][][] getEndStateProb() {
	return endStateProb;
}

public void setEndStateProb(double[][][] endStateProb) {
	this.endStateProb = endStateProb;
}






}
