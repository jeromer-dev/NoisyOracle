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

        // Correction de la boucle infinie (suppression du while)
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
                String[] values = line.split("\\s+");
                String[] transaction = new String[values.length];
                System.arraycopy(values, 0, transaction, 0, values.length);
                transactions.add(transaction);
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
        // C'est ici que l'erreur se produisait : on récupère la vraie taille N
        int nbTransactions = this.getTransactions().length; 
        List<DecisionRule> rules = new ArrayList<>();

        for (int i = 0; i < nbRules; i++) {
            int transactionIndex = random.nextInt(nbTransactions);
            String[] transaction = this.getTransactions()[transactionIndex];

            List<String> shuffledItems = new ArrayList<>(Arrays.asList(transaction));
            Collections.shuffle(shuffledItems);

            // CORRECTION CRITIQUE : Remplacer 100 par nbTransactions
            DecisionRule selectedDecisionRule = new DecisionRule(new HashSet<>(), "", this, nbTransactions, 1, smoothCounts,
                    measureNames);

            for (String item : shuffledItems) {
                if (getAntecedentItemsSet().contains(item)) {
                    if (random.Bernoulli(0.5))
                        selectedDecisionRule.addToX(item);
                } else {
                    selectedDecisionRule.setY(item);
                }
            }
            rules.add(selectedDecisionRule);
        }
        return rules;
    }
}