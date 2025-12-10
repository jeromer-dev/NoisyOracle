package tools.rules;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntConsumer;
import com.zaxxer.sparsebits.SparseBitSet;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative; // Import nécessaire
import tools.data.Dataset;
import tools.utils.AlternativeUtil;
import tools.utils.SetUtil;

@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
// AJOUT DE "implements IAlternative"
public class DecisionRule implements IRule, IAlternative { 
    
    private Dataset dataset;
    private Map<String, SparseBitSet> itemsMap;
    private @Setter int freqX, freqY, freqZ;
    private @Setter @Getter double smoothCounts;
    private @Setter @Getter String[] measureNames;
    private @Getter IAlternative alternative;
    private SparseBitSet coverX;
    private SparseBitSet coverY;
    private SparseBitSet coverZ;
    private Map<Set<String>, SparseBitSet> memoizedCoverX;
    private Map<Set<String>, SparseBitSet> memoizedCoverZ;
    private @Getter @Setter int maxSizeX, maxSizeZ;
    private CoverParallelCompute coverComputer;
    private String Y;
    private Set<String> itemsInX;
    private Set<String> itemsInZ;

    public DecisionRule(Set<String> itemsInX, String Y, Dataset dataset, int maxSizeX, int maxSizeZ,
            double smoothCounts, String[] measureNames) {
        this.dataset = dataset;
        this.itemsMap = this.dataset.getItemsMap();
        this.smoothCounts = smoothCounts;
        this.itemsInX = itemsInX;
        this.Y = Y;
        computeItemsInZ();
        setMaxSizeX(maxSizeX);
        setMaxSizeZ(maxSizeZ);
        initializeMemoization(getMaxSizeX(), getMaxSizeZ());
        this.coverComputer = new CoverParallelCompute(getDataset());
        computeNewCover(new String[] { "x", "y", "z" });
        updateFrequencies(new String[] { "x", "y", "z" });
        setMeasureNames(measureNames);
        this.alternative = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
    }
    
    // --- METHODES DE L'INTERFACE IAlternative ---
    
    @Override
    public double[] getVector() {
        return alternative.getVector();
    }

    @Override
    public double getOrderedValue(int i) {
        return alternative.getOrderedValue(i);
    }

    @Override
    public int[] getOrderedPermutation() {
        return alternative.getOrderedPermutation();
    }

    @Override
    public IAlternative deepCopy() {
        return alternative.deepCopy();
    }

    // --- LE RESTE DU CODE EXISTANT (Identique) ---

    public void expandSimpleCopy(DecisionRule originalRule) {
        this.dataset = originalRule.getDataset();
        this.itemsMap = getDataset().getItemsMap();
        computeItemsInZ();
        setMaxSizeX(originalRule.getMaxSizeX());
        setMaxSizeZ(originalRule.getMaxSizeZ());
        initializeMemoization(getMaxSizeX(), getMaxSizeZ());
        this.coverComputer = new CoverParallelCompute(getDataset());
        computeNewCover(new String[] { "x", "y", "z" });
        updateFrequencies(new String[] { "x", "y", "z" });
        setMeasureNames(originalRule.getMeasureNames());
        this.alternative = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
    }

    private void initializeMemoization(int maxSizeX, int maxSizeZ) {
        this.memoizedCoverX = new LinkedHashMap<Set<String>, SparseBitSet>(maxSizeX, 1.0f, true) {
            @Override protected boolean removeEldestEntry(Map.Entry<Set<String>, SparseBitSet> eldest) { return size() > maxSizeX; }
        };
        this.memoizedCoverZ = new LinkedHashMap<Set<String>, SparseBitSet>(maxSizeZ, 1.0f, true) {
            @Override protected boolean removeEldestEntry(Map.Entry<Set<String>, SparseBitSet> eldest) { return size() > maxSizeZ; }
        };
    }

    private void computeItemsInZ() {
        this.itemsInZ = new HashSet<>(this.itemsInX);
        if (!this.Y.isEmpty()) this.itemsInZ.add(this.Y);
    }

    private void updateFrequencies(String[] directions) {
        for (String direction : directions) {
            switch (direction) {
                case "x": updateFrequency(this.itemsInX, this.coverX, this::setFreqX); break;
                case "y": updateFrequency(this.Y, this.coverY, this::setFreqY); break;
                case "z": updateFrequency(this.itemsInZ, this.coverZ, this::setFreqZ); break;
            }
        }
    }

    private void updateFrequency(Set<?> items, SparseBitSet cover, IntConsumer frequencySetter) {
        frequencySetter.accept(items.isEmpty() ? 0 : cover.cardinality());
    }
    private void updateFrequency(String item, SparseBitSet cover, IntConsumer frequencySetter) {
        frequencySetter.accept(item.isEmpty() ? 0 : cover.cardinality());
    }

    private void updateAlternative() {
        this.alternative = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
    }

    private void computeNewCover(String[] directions) {
        for (String direction : directions) {
            switch (direction) {
                case "x": updateCoverX(); break;
                case "y": updateCoverY(); break;
                case "z": updateCoverZ(); break;
            }
        }
    }

    private void updateCoverX() {
        if (!this.itemsInX.isEmpty()) {
            SparseBitSet fromMemory = memoizedCoverX.get(this.itemsInX);
            if (fromMemory != null) {
                this.coverX = SetUtil.copyCover(fromMemory);
            } else {
                this.coverX = coverComputer.compute(itemsInX);
                memoizedCoverX.put(SetUtil.copySet(this.itemsInX), SetUtil.copyCover(this.coverX));
            }
        } else {
            this.coverX = SetUtil.createSparseBitSetAllOnes(this.dataset.getNbTransactions());
        }
    }

    private void updateCoverY() {
        if (!this.Y.isEmpty()) {
            SparseBitSet mapCover = this.itemsMap.get(this.Y);
            this.coverY = (mapCover != null) ? mapCover : new SparseBitSet();
        }
    }

    private void updateCoverZ() {
        if (!this.itemsInZ.isEmpty()) {
            SparseBitSet fromMemory = memoizedCoverZ.get(this.itemsInZ);
            if (fromMemory != null) {
                this.coverZ = SetUtil.copyCover(fromMemory);
            } else {
                this.coverZ = coverComputer.compute(itemsInZ);
                memoizedCoverZ.put(SetUtil.copySet(this.itemsInZ), SetUtil.copyCover(this.coverZ));
            }
        } else {
            this.coverZ = SetUtil.createSparseBitSetAllOnes(this.dataset.getNbTransactions());
        }
    }

    public void setX(Set<String> itemsInX) {
        this.itemsInX = itemsInX;
        computeItemsInZ();
        computeNewCover(new String[] { "x", "z" });
        updateFrequencies(new String[] { "x", "z" });
        updateAlternative();
    }

    public void addToX(String itemValue) {
        this.itemsInX.add(itemValue);
        SparseBitSet fromMemoryX = memoizedCoverX.get(this.itemsInX);
        if (fromMemoryX != null) {
            this.coverX = SetUtil.copyCover(fromMemoryX);
        } else {
            this.coverX.and(this.itemsMap.get(itemValue));
            memoizedCoverX.put(SetUtil.copySet(this.itemsInX), SetUtil.copyCover(this.coverX));
        }
        computeItemsInZ();
        SparseBitSet fromMemoryZ = memoizedCoverZ.get(this.itemsInZ);
        if (fromMemoryZ != null) {
            this.coverZ = SetUtil.copyCover(fromMemoryZ);
        } else {
            this.coverZ.and(this.itemsMap.get(itemValue));
            memoizedCoverZ.put(SetUtil.copySet(this.itemsInZ), SetUtil.copyCover(this.coverZ));
        }
        updateFrequencies(new String[] { "x", "z" });
        updateAlternative();
    }

    public void removeFromX(String itemValue) {
        if (this.itemsInX.contains(itemValue)) {
            this.itemsInX.remove(itemValue);
            computeItemsInZ();
            computeNewCover(new String[] { "x", "z" });
            updateFrequencies(new String[] { "x", "z" });
            updateAlternative();
        } else {
            throw new RuntimeException("Item index " + itemValue + " not found in item set X.");
        }
    }

    public void setY(String Y) {
        this.Y = Y;
        computeItemsInZ();
        computeNewCover(new String[] { "y", "z" });
        updateFrequencies(new String[] { "y", "z" });
        updateAlternative();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        DecisionRule rule = (DecisionRule) obj;
        if (Objects.equals(itemsInX, rule.itemsInX) && Objects.equals(Y, rule.Y)) {
            return true;
        }
        return Objects.equals(freqX, rule.getFreqX()) &&
               Objects.equals(freqY, rule.getFreqY()) &&
               Objects.equals(freqZ, rule.getFreqZ());
    }    

    @Override
    public int hashCode() {
        return Objects.hash(itemsInX, Y, freqX, freqY, freqZ);
    }
}