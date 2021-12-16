package br.ufrn.imd.selftraining.main;

import java.util.ArrayList;

import br.ufrn.imd.selftraining.core.Dataset;
import br.ufrn.imd.selftraining.core.SelfTraining;
import br.ufrn.imd.selftraining.core.SelfTrainingDws;
import br.ufrn.imd.selftraining.core.SelfTrainingEnsembleBased;
import br.ufrn.imd.selftraining.core.SelfTrainingEnsembleBasedTest;
import br.ufrn.imd.selftraining.core.SelfTrainingStandard;
import br.ufrn.imd.selftraining.filemanipulation.SelfTrainingOutputWriter;
import br.ufrn.imd.selftraining.results.SelfTrainingResult;

public class Main {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static SelfTraining st;
	public static ArrayList<Dataset> folds;
	public static SelfTrainingResult str;
	public static int seed;

	public static String selfTrainingStandard = "ST_VERSION_STANDARD";
	public static String selfTrainingRandom = "ST_RANDOM";

	public static String selfTrainingDwsc = "ST_VERSION_DWS_C";
	public static String selfTrainingDwscNewSelection = "ST_DWS_C_NEW_SELCTION";
	public static String selfTrainingDwscNewSelectionLabelling = "ST_DWS_C_NEW_SELCTION_LABELLING";
	public static String selfTrainingDwscNewLabelling = "ST_DWS_C_NEW_LABELLING";

	public static String selfTrainingEbalV1 = "ST_EBAL_V_01";
	public static String selfTrainingEbalV2 = "ST_EBAL_V_02";
	public static String selfTrainingEbalV3 = "ST_EBAL_V_03";

	public static String selfTrainingDwsaV1 = "ST_DWS_A_V_01";
	public static String selfTrainingDwsaV2 = "ST_DWS_A_V_02";

	public static SelfTrainingOutputWriter sow;
	public static String outputResultBasePath = "src/main/resources/results/";

	public static void main(String[] args) throws Exception {
		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		populateDatasets();

		seed = 19;

		for (Dataset d : datasets) {
			// run(d, selfTrainingDwsc);
			run(d, selfTrainingDwscNewSelection);
			run(d, selfTrainingDwscNewSelectionLabelling);
			run(d, selfTrainingDwscNewLabelling);
		}
	}

	public static void run(Dataset dataset, String selfTrainingVersion) throws Exception {

		System.out.println("Init " + selfTrainingVersion + " over " + dataset.getDatasetName() + " dataset");

		str = new SelfTrainingResult(numFolds, dataset.getDatasetName(), selfTrainingVersion);
		sow = new SelfTrainingOutputWriter(outputResultBasePath + selfTrainingVersion + "_" + dataset.getDatasetName());

		dataset.shuffleInstances(seed);
		folds = Dataset.splitDataset(dataset, numFolds);
		Dataset validation = new Dataset();
		str.setBegin(System.currentTimeMillis());

		System.out.print("CURRENT FOLD ... ");

		for (int i = 0; i < numFolds; i++) {
			System.out.print(i + 1);
			System.out.print(" ");

			validation = new Dataset(folds.get(i));
			ArrayList<Dataset> foldsForTest = new ArrayList<Dataset>();
			for (int j = 0; j < numFolds; j++) {
				if (i != j) {
					foldsForTest.add(folds.get(j));
				}
			}

			if (selfTrainingVersion.equals(selfTrainingDwscNewSelection)) {
				SelfTrainingDws stdws = new SelfTrainingDws(Dataset.joinDatasets(foldsForTest), validation);
				stdws.runStandardDwsNewSelection();
				st = (SelfTraining) stdws;
			} else if (selfTrainingVersion.equals(selfTrainingDwscNewSelectionLabelling)) {
				SelfTrainingDws stdws = new SelfTrainingDws(Dataset.joinDatasets(foldsForTest), validation);
				stdws.runStandardDwsNewSelectionLabelling();
				st = (SelfTraining) stdws;
			} else if (selfTrainingVersion.equals(selfTrainingDwscNewLabelling)) {
				SelfTrainingDws stdws = new SelfTrainingDws(Dataset.joinDatasets(foldsForTest), validation);
				stdws.runStandardDwsNewLabelling();
				st = (SelfTraining) stdws;
			} else if (selfTrainingVersion.equals(selfTrainingStandard)) {
				SelfTrainingStandard sts = new SelfTrainingStandard(Dataset.joinDatasets(foldsForTest), validation);
				sts.runStandard();
				st = (SelfTrainingStandard) sts;
			} else if (selfTrainingVersion.equals(selfTrainingRandom)) {
				SelfTrainingStandard sts = new SelfTrainingStandard(Dataset.joinDatasets(foldsForTest), validation);
				sts.runRandom();
				st = (SelfTrainingStandard) sts;
			} else if (selfTrainingVersion.equals(selfTrainingDwsc)) {
				SelfTrainingDws stdws = new SelfTrainingDws(Dataset.joinDatasets(foldsForTest), validation);
				stdws.runStandardDws();
				st = (SelfTraining) stdws;
			} else if (selfTrainingVersion.equals(selfTrainingEbalV1)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest),
						validation);
				steb.runEbalVersionOne();
				st = (SelfTrainingEnsembleBased) steb;
			} else if (selfTrainingVersion.equals(selfTrainingEbalV2)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest),
						validation);
				steb.runEbalVersionTwo();
				st = (SelfTrainingEnsembleBased) steb;
			} else if (selfTrainingVersion.equals(selfTrainingEbalV3)) {
				SelfTrainingEnsembleBasedTest stebt = new SelfTrainingEnsembleBasedTest(
						Dataset.joinDatasets(foldsForTest), validation);
				stebt.runTest();
				st = (SelfTrainingEnsembleBasedTest) stebt;
			} else if (selfTrainingVersion.equals(selfTrainingDwsaV1)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest),
						validation);
				steb.runDwsaVersionOne();
				st = (SelfTrainingEnsembleBased) steb;
			} else if (selfTrainingVersion.equals(selfTrainingDwsaV2)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest),
						validation);
				steb.runDwsaVersionTwo();
				st = (SelfTrainingEnsembleBased) steb;
			}

			str.setEnd(System.currentTimeMillis());
			str.addFoldResult(st.getResult());

			// sow.logDetailsAboutStep(dataset.getDatasetName(), i);
			// sow.addContentline(st.getHistory());
		}
		System.out.println();
		sow.addContentline(str.getResult());
		sow.writeInFile();

		str.showResult();
	}

	public static void populateDatasets() {
		String basePath = new String("src/main/resources/datasets/experiment_all/");

		ArrayList<String> sources = new ArrayList<String>();
		sources.add("Abalone.arff");
		sources.add("Adult.arff");
		sources.add("Arrhythmia.arff");
		sources.add("Automobile.arff");
		sources.add("Btsc.arff");
		sources.add("Car.arff");
		sources.add("Cnae.arff");
		sources.add("Dermatology.arff");
		sources.add("Ecoli.arff");
		sources.add("Flags.arff");
		sources.add("GermanCredit.arff");
		sources.add("Glass.arff");
		sources.add("Haberman.arff");
		sources.add("HillValley.arff");
		sources.add("Ilpd.arff");
		sources.add("ImageSegmentation_norm.arff");
		sources.add("KrVsKp.arff");
		sources.add("Leukemia.arff");
		sources.add("Madelon.arff");
		sources.add("MammographicMass.arff");
		sources.add("MultipleFeaturesKarhunen.arff");
		sources.add("Mushroom.arff");
		sources.add("Musk.arff");
		sources.add("Nursery.arff");
		sources.add("OzoneLevelDetection.arff");
		sources.add("PenDigits.arff");
		sources.add("PhishingWebsite.arff");
		sources.add("Pima.arff");
		sources.add("PlanningRelax.arff");
		// sources.add("Secom.arff");
		sources.add("Seeds.arff");
		sources.add("Semeion.arff");
		sources.add("SolarFlare.arff");
		sources.add("SolarFlare1.arff");
		sources.add("Sonar.arff");
		sources.add("SpectfHeart.arff");
		sources.add("TicTacToeEndgame.arff");
		sources.add("Twonorm.arff");
		sources.add("Vehicle.arff");
		sources.add("Waveform.arff");
		sources.add("Wilt.arff");
		sources.add("Wine.arff");
		sources.add("Yeast.arff");

		for (String s : sources) {
			Dataset d;
			try {
				d = new Dataset(basePath + s);
				datasets.add(d);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void populateDatasetsTest() {
		String basePath = new String("src/main/resources/datasets/test/");

		ArrayList<String> sources = new ArrayList<String>();
		sources.add("Iris.arff");
		// sources.add("Abalone.arff");

		for (String s : sources) {
			Dataset d;
			try {
				d = new Dataset(basePath + s);
				datasets.add(d);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

}
