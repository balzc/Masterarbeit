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
	
	private int[] actions = {0,1};
	private double[] prices;
	private double[] internalTemp;
	private double[] externalTemp;
	private double deltaPrice;
	private double deltaInternalTemp;
	private double deltaExternalTemp;
	private double minInternalTemp;
	private double maxInternalTemp;
	private double sdScale;
	private double delta_t;
	
	private double prefTemp;
	private double sensitivity;
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

	double tol = 0.0001;
	
	
	public HomeHeatingMDP(DoubleMatrix predMeanPrice, DoubleMatrix predCovPrice,DoubleMatrix predMeanExternalTemp, DoubleMatrix predCovExternalTemp, double deltaPrice,
			 int numSteps) {
		super();
		

		this.predMeanPrice = predMeanPrice;
		this.predCovPrice = predCovPrice;
		this.deltaPrice = deltaPrice;
		this.deltaExternalTemp = 2;
		this.deltaInternalTemp = 1;
		this.delta_t = 696;
		this.numSteps = numSteps;
		this.predCovExternalTemp = predCovExternalTemp;// predCovPrice;
		this.predCovInternalTemp = predCovExternalTemp;
		this.predMeanExternalTemp = predMeanExternalTemp;//predMeanPrice;
		this.predMeanInternalTemp = predMeanExternalTemp;
		this.sdScale = 5;
		this.sensitivity = 1;//0.5;
		this.powerOfHeater = 1000;
		this.prefTemp = 20;
		this.maxInternalTemp = 30;
		this.minInternalTemp = 10;
		this.massAir = 1205;//1205;
		this.coefficientOfPerformance = 2.5;
		this.leakageRate = 90;
		this.heatCapacity = 1000;//1000;
	}
	
	public void work(){
		System.out.println(rewards(20, 0, 20) + " " + rewards(20, 1, 20));
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
		double temp = internalTemp[j] + Q*delta_t/(massAir*heatCapacity);

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
		int numTemp = (int)(((maxInternalTemp - minInternalTemp))/(deltaInternalTemp))+1;
		this.internalTemp = new double[numTemp];
		for (int i = 0; i<numTemp; i++){
			internalTemp[i] = minInternalTemp + i*deltaInternalTemp;
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
	

	public double rewards(double internalTemp, int action, double price){
		return maxInternalTemp - sensitivity*(prefTemp - internalTemp)*(prefTemp - internalTemp) - action*price*delta_t*powerOfHeater/(1000.*3600.);
	}

	public int internalTempToState(double temp){
		int index = -1;
		double intTemp = roundToNextTemperature(temp, deltaInternalTemp);
		for(int i = 0; i<internalTemp.length; i++){

			if (Math.abs(internalTemp[i]-intTemp)<tol) {
				index = i;
				break;
			}
		}
		return index;
	}

	public int externalTempToState(double temp){
		int index = -1;
		double intTemp = roundToNextTemperature(temp, deltaExternalTemp);
		for(int i = 0; i<externalTemp.length; i++){
			if (Math.abs(externalTemp[i]-intTemp)<tol) {
				index = i;
				break;
			}
		}
		return index;
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

	public int findInternalTempProb(int iTempOld, int eTemp, int action){
		int index = -1;
		for(int i = 0; i<internalTemp.length; i++){
			if(Math.abs(internalTempProb[i][iTempOld][eTemp][action] - 1) < tol){
				index = i;
			}
		}
		return index;

	}
	

	public double roundToNextTemperature(double temp, double delta){
		return Math.floor((temp/delta + 0.5))*delta;
	}
	
	public double roundToNextPrice(double price, double delta){
		return Math.floor((price/delta + 0.5))*delta;
	}
	
	
	public double updateInternalTemperature(double internalTemp, double externalTemp, int heaterOn){
		double Q = heaterOn*powerOfHeater*coefficientOfPerformance - leakageRate*(internalTemp - externalTemp);
		System.out.println(Q);
		return internalTemp + Q*delta_t/(massAir*heatCapacity);
	}
	
	public void computeProbabilityTables(){
		if(PROFILING){
			time = System.nanoTime();
		}
		//priceProb[i][j][k] = Pr(price[i]| price[j], timestep = k+1)
		System.out.println(prices.length + " " + internalTemp.length + " " + externalTemp.length);
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
//					System.out.print(priceProb[i][j][k] + " ");
				}
//				System.out.println();
			}
//			System.out.println();

		}
//		System.out.println();
		//internal temp
		for (int i = 0; i<internalTemp.length;i++){
			for (int j = 0; j<internalTemp.length; j++){
				for (int k = 0; k<externalTemp.length; k++){
					for(int a = 0; a<actions.length; a++){
						internalTempProb[i][j][k][a] = computeInternalTempProb(i, j, k, a);
//						System.out.print(internalTempProb[i][j][k][a] + " ");

					}
//					System.out.println();

				}
//				System.out.println();

			}		
//			System.out.println();

		}
//		System.out.println();

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
//					System.out.print(externalTempProb[i][j][k] + " ");

				}
//				System.out.println();

			}
//			System.out.println();

		}
//		System.out.println();

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
							double qval = rewards(internalTemp[it],a,prices[p]);
//							System.out.println("action: " +a);
							double sum = 0;
							for(int pn = 0; pn < prices.length; pn++){
								for(int itn = 0; itn < internalTemp.length; itn++){
									for(int etn = 0; etn < externalTemp.length; etn++){
										qval += internalTempProb[itn][it][et][a]*externalTempProb[etn][et][t]*priceProb[p][pn][t]*qValues[t+1][pn][itn][etn];
										sum += internalTempProb[itn][it][et][a];
									}
								}
							}
//							System.out.println("sum: " +sum + " qval: " + qval);

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
		double sum = 0;
		for(int t = 0; t <numSteps; t++){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < internalTemp.length; it++){
					for(int et = 0; et < externalTemp.length; et++){
						System.out.print(prices[p] + " " + internalTemp[it] + " " + externalTemp[et] + " " + optPolicy[t][p][it][et] + "; ");
						sum += optPolicy[t][p][it][et];
					}
				}
			}
			System.out.println();
		}
		System.out.println("Sum: " + sum);
	}
	public void printQvals(){
		for(int t = 0; t <numSteps; t++){
			for(int p = 0; p < prices.length; p++){
				for(int it = 0; it < internalTemp.length; it++){
					for(int et = 0; et < externalTemp.length; et++){
						System.out.print(prices[p] + " " + internalTemp[it] + " " + externalTemp[et] + " " + qValues[t][p][it][et] + "; ");
					}
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

	public void printExtTemps(){
		for(int i = 0; i<externalTemp.length;i++){
			System.out.print(externalTemp[i] + " ");
		}
		System.out.println();

	}
	public void printIntTemps(){
		for(int i = 0; i<internalTemp.length;i++){
			System.out.print(internalTemp[i] + " ");
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

	public double[][] getExpectedRewards() {
		return expectedRewards;
	}

	public void setExpectedRewards(double[][] expectedRewards) {
		this.expectedRewards = expectedRewards;
	}

	public DoubleMatrix getPredMeanPrice() {
		return predMeanPrice;
	}

	public void setPredMeanPrice(DoubleMatrix predMeanPrice) {
		this.predMeanPrice = predMeanPrice;
	}

	public DoubleMatrix getPredCovPrice() {
		return predCovPrice;
	}

	public void setPredCovPrice(DoubleMatrix predCovPrice) {
		this.predCovPrice = predCovPrice;
	}

	public DoubleMatrix getPredMeanInternalTemp() {
		return predMeanInternalTemp;
	}

	public void setPredMeanInternalTemp(DoubleMatrix predMeanInternalTemp) {
		this.predMeanInternalTemp = predMeanInternalTemp;
	}

	public DoubleMatrix getPredCovInternalTemp() {
		return predCovInternalTemp;
	}

	public void setPredCovInternalTemp(DoubleMatrix predCovInternalTemp) {
		this.predCovInternalTemp = predCovInternalTemp;
	}

	public DoubleMatrix getPredMeanExternalTemp() {
		return predMeanExternalTemp;
	}

	public void setPredMeanExternalTemp(DoubleMatrix predMeanExternalTemp) {
		this.predMeanExternalTemp = predMeanExternalTemp;
	}

	public DoubleMatrix getPredCovExternalTemp() {
		return predCovExternalTemp;
	}

	public void setPredCovExternalTemp(DoubleMatrix predCovExternalTemp) {
		this.predCovExternalTemp = predCovExternalTemp;
	}

	public int[] getActions() {
		return actions;
	}

	public void setActions(int[] actions) {
		this.actions = actions;
	}

	public double[] getPrices() {
		return prices;
	}

	public void setPrices(double[] prices) {
		this.prices = prices;
	}

	public double[] getInternalTemp() {
		return internalTemp;
	}

	public void setInternalTemp(double[] internalTemp) {
		this.internalTemp = internalTemp;
	}

	public double[] getExternalTemp() {
		return externalTemp;
	}

	public void setExternalTemp(double[] externalTemp) {
		this.externalTemp = externalTemp;
	}

	public double getDeltaPrice() {
		return deltaPrice;
	}

	public void setDeltaPrice(double deltaPrice) {
		this.deltaPrice = deltaPrice;
	}

	public double getDeltaInternalTemp() {
		return deltaInternalTemp;
	}

	public void setDeltaInternalTemp(double deltaInternalTemp) {
		this.deltaInternalTemp = deltaInternalTemp;
	}

	public double getDeltaExternalTemp() {
		return deltaExternalTemp;
	}

	public void setDeltaExternalTemp(double deltaExternalTemp) {
		this.deltaExternalTemp = deltaExternalTemp;
	}

	public double getMinInternalTemp() {
		return minInternalTemp;
	}

	public void setMinInternalTemp(double minInternalTemp) {
		this.minInternalTemp = minInternalTemp;
	}

	public double getMaxInternalTemp() {
		return maxInternalTemp;
	}

	public void setMaxInternalTemp(double maxInternalTemp) {
		this.maxInternalTemp = maxInternalTemp;
	}

	public double getSdScale() {
		return sdScale;
	}

	public void setSdScale(double sdScale) {
		this.sdScale = sdScale;
	}

	public double getPrefTemp() {
		return prefTemp;
	}

	public void setPrefTemp(double prefTemp) {
		this.prefTemp = prefTemp;
	}

	public double getSensitivity() {
		return sensitivity;
	}

	public void setSensitivity(double sensitivity) {
		this.sensitivity = sensitivity;
	}

	public double getPowerOfHeater() {
		return powerOfHeater;
	}

	public void setPowerOfHeater(double powerOfHeater) {
		this.powerOfHeater = powerOfHeater;
	}

	public double getCoefficientOfPerformance() {
		return coefficientOfPerformance;
	}

	public void setCoefficientOfPerformance(double coefficientOfPerformance) {
		this.coefficientOfPerformance = coefficientOfPerformance;
	}

	public double getLeakageRate() {
		return leakageRate;
	}

	public void setLeakageRate(double leakageRate) {
		this.leakageRate = leakageRate;
	}

	public double getMassAir() {
		return massAir;
	}

	public void setMassAir(double massAir) {
		this.massAir = massAir;
	}

	public double getHeatCapacity() {
		return heatCapacity;
	}

	public void setHeatCapacity(double heatCapacity) {
		this.heatCapacity = heatCapacity;
	}

	public boolean isPROFILING() {
		return PROFILING;
	}

	public void setPROFILING(boolean pROFILING) {
		PROFILING = pROFILING;
	}

	public double getTime() {
		return time;
	}

	public void setTime(double time) {
		this.time = time;
	}

	public double[][][] getPriceProb() {
		return priceProb;
	}

	public void setPriceProb(double[][][] priceProb) {
		this.priceProb = priceProb;
	}

	public double[][][][] getInternalTempProb() {
		return internalTempProb;
	}

	public void setInternalTempProb(double[][][][] internalTempProb) {
		this.internalTempProb = internalTempProb;
	}

	public double[][][] getExternalTempProb() {
		return externalTempProb;
	}

	public void setExternalTempProb(double[][][] externalTempProb) {
		this.externalTempProb = externalTempProb;
	}

	public int getNumSteps() {
		return numSteps;
	}

	public void setNumSteps(int numSteps) {
		this.numSteps = numSteps;
	}


}
