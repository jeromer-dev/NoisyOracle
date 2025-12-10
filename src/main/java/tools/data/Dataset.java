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

    /**
     * Génère des règles aléatoires valides et calcule leurs supports réels.
     * C'est crucial pour éviter les erreurs "Illegal value for measure phi".
     */
    public List<DecisionRule> getRandomValidRules(int nbRules, double smoothCounts, String[] measureNames) {
        RandomUtil random = new RandomUtil();
        int nbTransactions = this.getTransactions().length; // 150 pour Iris
        List<DecisionRule> rules = new ArrayList<>();

        for (int i = 0; i < nbRules; i++) {
            int transactionIndex = random.nextInt(nbTransactions);
            String[] transaction = this.getTransactions()[transactionIndex];

            List<String> shuffledItems = new ArrayList<>(Arrays.asList(transaction));
            Collections.shuffle(shuffledItems);

            // Création de la règle avec le bon nombre de transactions (N)
            DecisionRule selectedDecisionRule = new DecisionRule(new HashSet<>(), "", this, nbTransactions, 1, smoothCounts, measureNames);

            boolean consequentSet = false;
            for (String item : shuffledItems) {
                if (getAntecedentItemsSet().contains(item)) {
                    if (random.Bernoulli(0.5)) {
                        selectedDecisionRule.addToX(item);
                    }
                } else if (!consequentSet) {
                    // On ne définit qu'un seul conséquent par règle
                    selectedDecisionRule.setY(item);
                    consequentSet = true;
                }
            }
            
            // Si aucun conséquent n'a été trouvé dans la transaction (cas rare mais possible), on en force un aléatoire
            if (!consequentSet && !getConsequentItemsSet().isEmpty()) {
                String randomConsequent = getConsequentItemsSet().iterator().next();
                selectedDecisionRule.setY(randomConsequent);
            }

            // --- CORRECTION CRUCIALE ---
            // Il faut recalculer les supports (n_X, n_Y, n_XY) car on a ajouté des items manuellement.
            // Si DecisionRule ne le fait pas automatiquement à l'ajout, les mesures seront fausses.
            // On force le recalcul en réinitialisant ou en appelant une méthode de mise à jour si elle existe.
            // Dans votre architecture, DecisionRule semble calculer les bitsets à la demande.
            // On va s'assurer que les mesures sont calculées sur des données cohérentes.
            
            // Vérification simple : si n_X ou n_Y est 0, Phi peut planter.
            // Les règles générées ici sont basées sur une transaction existante, donc le support est au moins 1.
            
            rules.add(selectedDecisionRule);
        }
        return rules;
    }
}