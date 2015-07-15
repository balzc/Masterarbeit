package mdp;

import java.util.Arrays;

import org.apache.commons.math3.special.Erf;
import org.jblas.DoubleMatrix;

import main.Main;
import gp.GP;

public class EVMDPOld {
	private int[][][] optPolicy;
	private double[][][] qValues;
	private DoubleMatrix predMeanPrice;
	private DoubleMatrix predCovPrice;

	private int[] actions = {0,1};
	private double[] prices;
	private double[] loads;
	private double deltaPrice;
	private double qMax;
	private double qRequired;
	private double qSlope;
	private double vMin;
	private double tStart;
	private double tCrit;
	private double sdScale;

	private boolean PROFILING = false;
	private double time;
	private double[][][] priceProb;
	private double[][][] loadProb;
	private int numSteps;

	private double wattPerUnit = 1;
	
	double tol = 0.0001;


	public EVMDPOld(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,	int numSteps) {


		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.numSteps = numSteps;
		this.sdScale = deltaPrice;
		this.qMax = 2;
		this.qRequired = 1;
		this.qSlope = 10*wattPerUnit;
		this.tStart = 5;
		this.tCrit = 6;
		this.vMin = 10*wattPerUnit*qRequired;
		
		// deltat = (tcrit-tplug)/numsteps
	}

	public EVMDPOld(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice, double deltaPrice,	int numSteps, double qMax, double qRequired, double qSlope, double tStart, double tCrit, double vMin) {


		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.numSteps = numSteps;
		this.sdScale = deltaPrice;
		this.qMax = qMax;
		this.qRequired = qRequired;
		this.qSlope = qSlope;
		this.tStart = tStart;
		this.tCrit = tCrit;
		this.vMin = vMin;
		
		// deltat = (tcrit-tplug)/numsteps
	}
	
	
	public void work(){
		System.out.println("rewardtest " + (rewards(1, 0, 21,5)) + " " + value(0, 2) + " " + value(1, 2));
		
		computePrices();


		computeLoad();
		computeProbabilityTables();
		printLoads();
		solveMDP();
		printQvals();
//		printOptPolicy();
	}

	//probability table: priceProb[i][j][k] = Pr(prices[i] | prices[j], timestep=k)
	public double computePriceProb(int i, int j, int k){
		double m = predMeanPrice.get(k) + predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1)*(prices[j] - predMeanPrice.get(k-1));
		double s = Math.sqrt(predCovPrice.get(k,k) - predCovPrice.get(k,k-1)*predCovPrice.get(k,k-1)/predCovPrice.get(k-1,k-1));

		double y1 = prices[i] - 0.5*deltaPrice;
		double y2 = prices[i] + 0.5*deltaPrice;
		return 0.5*(Erf.erf( (y1-m)/(Math.sqrt(2)*s), (y2-m)/(Math.sqrt(2)*s)) );
		//		if(prices[i] == 40 && (k < 24 ) ){
		//			return 1;
		//		}else if(prices[i] == 40 && (k < 48 && k > 23) ){
		//			return 1;
		//		} else if(prices[i] == 0 && (k < 96 && k >= 48) ){
		//			return 1;
		//		} else {
		//			return 0;
		//		}

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
		this.loads = new double[(int)qMax+1];
		for (int i = 0; i<qMax+1; i++){
			loads[i] = i*wattPerUnit;
		}
	}



	// Tested, Works as intended
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


	public double rewards(double q, int action, double price, int t){
		double cost = price * action * wattPerUnit;
//		if(value(q+action * wattPerUnit,t+1) >= value(q,t)){
			return value(q+action * wattPerUnit,t+1)-value(q,t)-cost;
//		}
//		else{
//			return 0;
//		}

	}

	public double value(double qin, int t){
		double q = Math.min(qin, qMax);
		double requiredLoad = qRequired*wattPerUnit;
		double qflex = Math.max((q-requiredLoad), 0);
		if(q >= qRequired*wattPerUnit){
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
		if(initialLoad + action*wattPerUnit > qMax*wattPerUnit){
			return qMax*wattPerUnit;
		}
		else{
			return initialLoad + action*wattPerUnit;
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
			for(int q = 0; q < loads.length; q++){


				double temp = rewards(loads[q],0,prices[p],numSteps-1);
				qValues[numSteps-1][p][q] = temp;



			}
		}


		for(int t = numSteps-2; t >-1; t--){
			for(int p = 0; p < prices.length; p++){
				for(int q = 0; q < loads.length; q++){
					double currentMax = Double.NEGATIVE_INFINITY;
					int currentBestAction = 0;
					int counter[] = {0,0};

					for(int a = 0; a < actions.length; a++){
						double qval = rewards(loads[q],a,prices[p],t);
						//							System.out.println("action: " +a);
						double sumextt = 0;
						double sumintt = 0;
						double sump = 0;
						double sumtot = 0;
						for(int pn = 0; pn < prices.length; pn++){
							for(int qn = 0; qn < loads.length; qn++){
								qval += loadProb[qn][q][a]*priceProb[pn][p][t]*qValues[t+1][pn][qn];
							}
						}						
						if(qval > currentMax){
							currentMax = qval;
							currentBestAction = a;
							qValues[t][p][q] = qval;
							optPolicy[t][p][q] = currentBestAction;



						} 
					}


				}
			}
		}


	}

	public void printOptPolicy(){
		double sum = 0;
		for(int t = 0; t <numSteps; t++){
			System.out.print(t + "  ");

			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < loads.length; it++){
					System.out.print(prices[p] + " " + loads[it] + " "  + optPolicy[t][p][it] + "; ");
					sum += optPolicy[t][p][it];

				}
				System.out.println();

			}
		}
		System.out.println("Sum: " + sum);
	}

public void printQvals(){
	for(int t = 0; t <numSteps; t++){
		for(int p = 0; p < prices.length; p++){
			for(int it = 0; it < loads.length; it++){
				if(qValues[t][p][it] != 0){
					System.out.print(t + " "+ prices[p] + " " + loads[it] + " " +  qValues[t][p][it]+ "; ");
				}
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
