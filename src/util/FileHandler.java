package util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

public class FileHandler {



	public static void matrixToCsv(DoubleMatrix m, String destination){
		try
		{		
			FileWriter writer = new FileWriter(destination);

			for(int i = 0; i < m.rows; i++){
				for(int j = 0; j < m.columns; j++){
					writer.append(""+m.get(i,j));
					writer.append(',');
				}
				writer.append('\n');

			}
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static void safeDailyReport(DoubleMatrix m, String destination, double vmintrue, double mqtrue){
		try
		{		
			FileWriter writer = new FileWriter(destination);
			writer.append("Daily Utilities,,,,");
			writer.append("Daily Costs,,,,");
			writer.append("Daily Loads,,,,");
			writer.append("vmin," + vmintrue + ",mq," + mqtrue);
			writer.append('\n');
			writer.append("MDP,LPL,PAFL,SML,");
			writer.append("MDP,LPL,PAFL,SML,");
			writer.append("MDP,LPL,PAFL,SML,");
			writer.append('\n');

			for(int i = 0; i < m.rows; i++){
				for(int j = 0; j < m.columns; j++){
					writer.append(""+m.get(i,j));
					writer.append(',');
				}
				writer.append('\n');

			}
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static void safeTimeStepReport(DoubleMatrix m, String destination){
		try
		{		
			FileWriter writer = new FileWriter(destination);
			writer.append("Actions,,,,");
			writer.append("Loads,,,,");
			writer.append("Costs,,,,");
			writer.append("Predicted Prices,");
			writer.append("Actual Prices,");
			writer.append("Timestep,");

			writer.append('\n');
			writer.append("MDP,LPL,PAFL,SML,");
			writer.append("MDP,LPL,PAFL,SML,");
			writer.append("MDP,LPL,PAFL,SML,,,,");
			writer.append('\n');

			for(int i = 0; i < m.rows; i++){
				for(int j = 0; j < m.columns; j++){
					writer.append(""+m.get(i,j));
					writer.append(',');
				}
				writer.append('\n');

			}
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static void safeStoppingReport(DoubleMatrix m, String destination){
		try
		{		
			FileWriter writer = new FileWriter(destination);
			writer.append("Average Regret,");
			writer.append("mq,");
			writer.append("vmin,");

			writer.append('\n');
		

			for(int i = 0; i < m.rows; i++){
				for(int j = 0; j < m.columns; j++){
					writer.append(""+m.get(i,j));
					writer.append(',');
				}
				writer.append('\n');

			}
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static DoubleMatrix csvToMatrix(String filename){
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		DoubleMatrix result = new DoubleMatrix();
		ArrayList<String[]> res = new ArrayList();
		try {

			br = new BufferedReader(new FileReader(filename));
			int cellCounter = 0;
			int lineCounter = 0;
			while ((line = br.readLine()) != null) {
				String[] splitLine = line.split(cvsSplitBy);
				res.add(splitLine);
				lineCounter++;
				cellCounter = splitLine.length;
			}
			result = new DoubleMatrix(lineCounter, cellCounter);

			for(int sray = 0; sray < res.size(); sray++){
				for(int s = 0;s < res.get(sray).length; s++){
					if(res.get(sray)[s].matches("-?\\d+(\\.\\d+)?")){
						result.put(sray,s, Double.valueOf(res.get(sray)[s]));
					}
				}
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return result;
	}
	

	public static void resultsToFile(String content, String destination){
		try
		{		
			FileWriter writer = new FileWriter(destination);
			writer.append(content);
		
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}

}
