package br.ufrn.imd.selftraining.main;

import java.util.ArrayList;

import br.ufrn.imd.selftraining.core.Dataset;
import br.ufrn.imd.selftraining.core.SelfTrainingMachine;
import br.ufrn.imd.selftraining.filemanipulation.SelfTrainingOutputWriter;
import br.ufrn.imd.selftraining.results.SelfTrainingResult;

public class Main {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static SelfTrainingMachine stm;
	public static ArrayList<Dataset> folds;
	public static SelfTrainingResult str;
	public static int seed;

	public static String selfTrainingVersionOne = "ST_VERSION_01";
	public static String selfTrainingVersionTwo = "ST_VERSION_02";
	public static String selfTrainingStandard = "ST_VERSION_STANDARD_LAZY";
	public static String selfTrainingStandard2 = "ST_VERSION_STANDARD";
	
	public static SelfTrainingOutputWriter sow;
	public static String outputResultBasePath = "src/main/resources/results/";
	
	
	public static void main(String[] args) throws Exception {
		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		populateDatasets();

		seed = 19;
		
		for (Dataset d : datasets) {
			
			run(d, selfTrainingVersionOne);
			run(d, selfTrainingVersionTwo);
			run(d, selfTrainingStandard);
			run(d, selfTrainingStandard2);
		}
	}

	public static void populateDatasets() {
		String basePath = new String("src/main/resources/datasets/experiment2/");
		
		ArrayList<String> sources = new ArrayList<String>();
		sources.add("Btsc.arff");
		sources.add("Car.arff");
		sources.add("Cnae-9.arff");
		sources.add("Haberman.arff");
		sources.add("Hill-valley.arff");
		sources.add("Ilpd.arff");
		sources.add("Image-segmentation.arff");
		sources.add("Kr-vs-kp.arff");
		sources.add("Leukemia.arff");
		sources.add("Mammographic-mass.arff");
		sources.add("Multiple-features-karhunen.arff");
		sources.add("Mushroom.arff");
		sources.add("Musk.arff");
		sources.add("Ozone-level-detection.arff");
		sources.add("Pen-digits.arff");
		sources.add("Phishing-website.arff");
		sources.add("Pima.arff");
		sources.add("Planning-relax.arff");
		sources.add("Seeds.arff");
		sources.add("Semeion.arff");
		sources.add("Solar-flare.arff");
		sources.add("Spectf-heart.arff");
		sources.add("Tic-tac-toe-endgame.arff");
		sources.add("Twonorm.arff");
		sources.add("Vehicle.arff");
		sources.add("Waveform.arff");
		sources.add("Wilt.arff");
		
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
		str.setBegin(System.currentTimeMillis());
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
			else if(selfTrainingVersion.equals(selfTrainingStandard2)) {
				stm.runStandard2();
			}
			
			str.setEnd(System.currentTimeMillis());
			str.addFoldResult(stm.getResult());
			
			sow.logDetailsAboutStep(dataset.getDatasetName(), i);
			sow.addContentline(stm.getHistory());
		}
		sow.addContentline(str.getResult());
		sow.writeInFile();
		
		str.showResult();
	}
	

}
