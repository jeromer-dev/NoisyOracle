package tools.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.zaxxer.sparsebits.SparseBitSet;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;

@Getter
public class Dataset {

    private String filename;
    private String expDir;
    private @Setter @Getter Map<String, SparseBitSet> itemsMap; 
    private @Setter @Getter Set<String> consequentItemsSet; 
    private @Setter @Getter Set<String> antecedentItemsSet; 
    private @Setter @Getter String[] consequentItemsArray; 
    private @Setter @Getter String[] antecedentItemsArray; 
    private @Setter @Getter int nbAntecedentItems; 
    private @Setter @Getter int nbConsequentItems; 
    private @Setter @Getter int nbTransactions; 
    private @Setter @Getter String[][] transactions; 
    private @Getter UnionFind equivalenceClasses; 

    public Dataset(String filename, String expDir, Set<String> consequentItemsSet) throws IOException {
        this.filename = filename;
        this.expDir = expDir;
        this.transactions = getTransactionalDataset();
        this.nbTransactions = this.transactions.length;
        getItemsFromTransactions();
        setConsequentItemsSet(consequentItemsSet);
        setConsequentItemsArray(getConsequentItemsSet().toArray(new String[0]));
        initializeAntecedentItemsValues();
        this.nbAntecedentItems = getAntecedentItemsSet().size();
        this.nbConsequentItems = getConsequentItemsSet().size();
        findEquivalenceClasses();
    }

    public Dataset(String[][] transactionalDataset, Set<String> consequentItemsSet) {
        this.transactions = transactionalDataset;
        this.nbTransactions = transactionalDataset.length;
        getItemsFromTransactions();
        setConsequentItemsSet(consequentItemsSet);
        setConsequentItemsArray(getConsequentItemsSet().toArray(new String[0]));
        initializeAntecedentItemsValues();
        this.nbAntecedentItems = getAntecedentItemsSet().size();
        this.nbConsequentItems = getConsequentItemsSet().size();
        findEquivalenceClasses();
    }

    public void findEquivalenceClasses() {
        HashSet<String> allItems = new HashSet<>();
        allItems.addAll(getConsequentItemsSet());
        allItems.addAll(getAntecedentItemsSet());
        this.equivalenceClasses = new UnionFind(allItems.toArray(new String[0]));
        String[][] transactions = getTransactions();
        for (String[] transaction : transactions) {
            if (transaction.length > 0) {
                String classRep = transaction[0];
                for (String item : transaction) {
                    equivalenceClasses.union(item, classRep);
                }
            }
        }
    }

    private void initializeAntecedentItemsValues() {
        setAntecedentItemsSet(getItemsMap().keySet().stream()
                .filter(key -> !getConsequentItemsSet().contains(key))
                .collect(Collectors.toSet()));
        setAntecedentItemsArray(getAntecedentItemsSet().toArray(new String[0]));
    }

    public String[][] getTransactionalDataset() throws IOException {
        List<String[]> transactions = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(expDir + filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] values = line.split("[,\\s]+");
                List<String> validValues = new ArrayList<>();
                for (String v : values) if (!v.isEmpty()) validValues.add(v.trim());
                if(!validValues.isEmpty()) transactions.add(validValues.toArray(new String[0]));
            }
        }
        String[][] transactionArray = new String[transactions.size()][];
        for (int i = 0; i < transactions.size(); i++) {
            transactionArray[i] = transactions.get(i);
        }
        return transactionArray;
    }

    public void getItemsFromTransactions() {
        this.itemsMap = new HashMap<>();
        for (int transactionIndex = 0; transactionIndex < transactions.length; transactionIndex++) {
            String[] transaction = this.transactions[transactionIndex];
            for (String itemValue : transaction) {
                itemsMap.computeIfAbsent(itemValue, k -> new SparseBitSet()).set(transactionIndex);
            }
        }
    }

    public List<DecisionRule> getRandomValidRules(int nbRules, double smoothCounts, String[] measureNames) {
        RandomUtil random = new RandomUtil();
        int nbTransactions = this.getTransactions().length;
        List<DecisionRule> rules = new ArrayList<>();
        
        int attempts = 0;
        int maxAttempts = nbRules * 200;

        while (rules.size() < nbRules && attempts < maxAttempts) {
            attempts++;
            int transactionIndex = random.nextInt(nbTransactions);
            String[] transaction = this.getTransactions()[transactionIndex];

            List<String> shuffledItems = new ArrayList<>(Arrays.asList(transaction));
            Collections.shuffle(shuffledItems);

            String selectedConsequent = null;
            for (String item : shuffledItems) {
                if (getConsequentItemsSet().contains(item)) {
                    selectedConsequent = item;
                    break;
                }
            }
            if (selectedConsequent == null) continue;

            DecisionRule selectedDecisionRule = new DecisionRule(new HashSet<>(), selectedConsequent, this, nbTransactions, 1, smoothCounts, measureNames);

            for (String item : shuffledItems) {
                if (getAntecedentItemsSet().contains(item) && !item.equals(selectedConsequent)) {
                    if (random.Bernoulli(0.5)) selectedDecisionRule.addToX(item);
                }
            }
            
            // CORRECTION CRITIQUE : Éliminer les règles triviales (0% ou 100% de couverture)
            // Cela évite les divisions par zéro dans le calcul de Phi (Chi-Carré)
            int fZ = selectedDecisionRule.getFreqZ();
            int fX = selectedDecisionRule.getFreqX();
            int fY = selectedDecisionRule.getFreqY();
            
            if (fZ > 0 && fZ < nbTransactions && fX > 0 && fX < nbTransactions && fY > 0 && fY < nbTransactions) {
                rules.add(selectedDecisionRule);
            }
        }
        return rules;
    }
}