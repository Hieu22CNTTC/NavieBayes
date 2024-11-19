import java.io.*;
import java.util.*;

public class NaiveBayesClassifier {

    // Lưu trữ dữ liệu đọc từ file
    static class DataPoint {
        double x, y;
        String label;

        DataPoint(double x, double y, String label) {
            this.x = x;
            this.y = y;
            this.label = label;
        }
    }

    public static void main(String[] args) throws IOException {
        // Đọc dữ liệu từ file
        List<DataPoint> dataPoints = readDataFromFile("data.txt");

        // Các điểm cần dự đoán
        DataPoint S9 = new DataPoint(7.5, 2.0, "?");
        DataPoint S10 = new DataPoint(4, 6, "?");

        // Dự đoán lớp cho S9 và S10
        System.out.println("S9 thuoc lop: " + predictClass(S9, dataPoints));
        System.out.println("S10 thuoc lop: " + predictClass(S10, dataPoints));
    }

    // Đọc dữ liệu từ file
    private static List<DataPoint> readDataFromFile(String fileName) throws IOException {
        List<DataPoint> dataPoints = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String line;

        // Bỏ qua dòng đầu tiên (header)
        br.readLine();

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);
            String label = parts[2];
            dataPoints.add(new DataPoint(x, y, label));
        }
        br.close();
        return dataPoints;
    }

    // Hàm dự đoán lớp
    private static String predictClass(DataPoint s, List<DataPoint> dataPoints) {
        Map<String, Integer> labelCounts = new HashMap<>();
        Map<String, Double> meanX = new HashMap<>();
        Map<String, Double> meanY = new HashMap<>();
        Map<String, Double> varianceX = new HashMap<>();
        Map<String, Double> varianceY = new HashMap<>();

        // Tính các thông số (mean và variance) cho từng lớp
        for (DataPoint dp : dataPoints) {
            labelCounts.put(dp.label, labelCounts.getOrDefault(dp.label, 0) + 1);
            meanX.put(dp.label, meanX.getOrDefault(dp.label, 0.0) + dp.x);
            meanY.put(dp.label, meanY.getOrDefault(dp.label, 0.0) + dp.y);
        }

        for (String label : labelCounts.keySet()) {
            meanX.put(label, meanX.get(label) / labelCounts.get(label));
            meanY.put(label, meanY.get(label) / labelCounts.get(label));
        }

        for (DataPoint dp : dataPoints) {
            varianceX.put(dp.label, varianceX.getOrDefault(dp.label, 0.0) + Math.pow(dp.x - meanX.get(dp.label), 2));
            varianceY.put(dp.label, varianceY.getOrDefault(dp.label, 0.0) + Math.pow(dp.y - meanY.get(dp.label), 2));
        }

        for (String label : labelCounts.keySet()) {
            varianceX.put(label, varianceX.get(label) / labelCounts.get(label));
            varianceY.put(label, varianceY.get(label) / labelCounts.get(label));
        }

        // Tính xác suất theo công thức Naive Bayes
        String bestLabel = null;
        double bestProbability = -1;

        for (String label : labelCounts.keySet()) {
            double probability = labelCounts.get(label) / (double) dataPoints.size();

            // Gaussian probability density function
            probability *= gaussian(s.x, meanX.get(label), varianceX.get(label));
            probability *= gaussian(s.y, meanY.get(label), varianceY.get(label));

            if (probability > bestProbability) {
                bestProbability = probability;
                bestLabel = label;
            }
        }

        return bestLabel;
    }

    // Hàm tính Gaussian
    private static double gaussian(double value, double mean, double variance) {
        double exponent = Math.exp(-Math.pow(value - mean, 2) / (2 * variance));
        return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
    }
}
