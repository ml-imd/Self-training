package br.ufrn.imd.selftraining;

import java.util.ArrayList;

import br.ufrn.imd.selftraining.filemanipulation.SelfTrainingOutputWriter;

public class Main {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static SelfTrainingMachine stm;
	public static ArrayList<Dataset> folds;
	public static SelfTrainingResult str;
	public static int seed;

	public static String selfTrainingVersionOne = "ST_VERSION_01";
	public static String selfTrainingVersionTwo = "ST_VERSION_02";
	public static String selfTrainingStandard = "ST_VERSION_STANDARD";
	
	public static SelfTrainingOutputWriter sow;
	public static String outputResultBasePath = "src/main/resources/results/";
	
	
	public static void main(String[] args) throws Exception {
		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		populateDatasets();

		seed = 19;
		
		for (Dataset d : datasets) {
			
			//run(d, selfTrainingVersionOne);
			//run(d, selfTrainingVersionTwo);
			run(d, selfTrainingStandard);
			
		}
	}

	public static void populateDatasets() {
		String basePath = new String("src/main/resources/datasets/");
		
		ArrayList<String> sources = new ArrayList<String>();
		//sources.add("Abalone.arff");
		//sources.add("Adult.arff");
		//sources.add("Arrhythmia.arff");
		sources.add("Automobile.arff");
		//sources.add("Car.arff");
		//sources.add("Dermatology.arff");
		//sources.add("Ecoli.arff");
		//sources.add("Flags.arff");
		//sources.add("GermanCredit.arff");
		//sources.add("GlassIdentification.arff");
		//sources.add("ImageSegmentation.arff");
		//sources.add("KR-vs-KP.arff");
		//sources.add("Madelon.arff");
		//sources.add("Nursery.arff");
		//sources.add("Secom.arff");
		//sources.add("Semeion.arff");
		//sources.add("SolarFlare1.arff");
		//sources.add("Sonar.arff");
		//sources.add("Waveform.arff");
		//sources.add("Wine.arff");
		//sources.add("Yeast.arff");
		
		for(String s: sources) {
			Dataset d;
			try {
				d = new Dataset(basePath+s);
				datasets.add(d);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
	}

	public static void run(Dataset dataset, String selfTrainingVersion) throws Exception {
		
		str = new SelfTrainingResult(numFolds, dataset.getDatasetName(), selfTrainingVersion);
		sow = new SelfTrainingOutputWriter(outputResultBasePath + selfTrainingVersion + "_" + dataset.getDatasetName());
		
		dataset.shuffleInstances(seed);
		folds = Dataset.splitDataset(dataset, numFolds);
		Dataset validation = new Dataset();
		
		for (int i = 0; i < numFolds; i++) {
			
			validation = new Dataset(folds.get(i));
			ArrayList<Dataset> foldsForTest = new ArrayList<Dataset>();
			for (int j = 0; j < numFolds; j++) {
				if (i != j) {
					foldsForTest.add(folds.get(j));
				}
			}
			
			stm = new SelfTrainingMachine(Dataset.joinDatasets(foldsForTest), validation);
			
			if(selfTrainingVersion.equals(selfTrainingVersionOne)) {
				stm.runVersionOne();
			}
			else if(selfTrainingVersion.equals(selfTrainingVersionTwo)) {
				stm.runVersionTwo();
			}
			else if(selfTrainingVersion.equals(selfTrainingStandard)) {
				stm.runStandard();
			}
			
			str.addFoldResult(stm.getResult());
			
			sow.logDetailsAboutStep(dataset.getDatasetName(), i);
			sow.addContentline(stm.getHistory());
		}
		sow.addContentline(str.getResult());
		sow.writeInFile();
		
		str.showResult();
	}
	

}
