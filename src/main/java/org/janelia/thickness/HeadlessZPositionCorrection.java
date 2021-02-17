package org.janelia.thickness;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.DoubleType;
import org.janelia.thickness.inference.InferFromMatrix;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.inference.fits.AbstractCorrelationFit;
import org.janelia.thickness.inference.fits.GlobalCorrelationFitAverage;
import org.janelia.thickness.inference.fits.LocalCorrelationFitAverage;
import org.janelia.thickness.inference.visitor.LazyVisitor;
import org.janelia.thickness.inference.visitor.Visitor;
import org.janelia.thickness.plugin.RealSumFloatNCC;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.janelia.thickness.plugin.ZPositionCorrection.createEmptyMatrix;
import static org.janelia.thickness.plugin.ZPositionCorrection.wrapDouble;

public class HeadlessZPositionCorrection {

    /**
     * Runs {@link InferFromMatrix#estimateZCoordinates} in "headless" mode without any visualization.
     *
     * Core logic was copied from {@link org.janelia.thickness.plugin.ZPositionCorrection#run}.
     *
     * @param  slicePaths       z-ordered file system paths of slice images to process.
     * @param  baseOptions      options for estimation.
     * @param  nLocalEstimates  number of local estimates used to derive estimateWindowRadius option value.
     *
     * @return estimated z coordinates for each specified slice.
     *
     * @throws IllegalArgumentException
     *   if there are too few or too many slices to process.
     */
    public static double[] estimateZCoordinates(final List<String> slicePaths,
                                                final Options baseOptions,
                                                final int nLocalEstimates)
            throws IllegalArgumentException {

        final int nSlices = slicePaths.size();

        // clone baseOptions since we need to override/derive some values
        final Options options = baseOptions.clone();

        options.estimateWindowRadius = nSlices / nLocalEstimates;

        // override comparisonRange if we only have a small number of slices
        if (nSlices < options.comparisonRange) {
            options.comparisonRange = nSlices;
        }

        final RandomAccessibleInterval<DoubleType> matrix = buildNCCMatrix(slicePaths, options.comparisonRange);

        final double[] startingCoordinates = new double[(int) matrix.dimension(0)];
        for (int i = 0; i < startingCoordinates.length; i++) {
            startingCoordinates[i] = i;
        }

        final AbstractCorrelationFit correlationFit =
                nLocalEstimates < 2 ?
                new GlobalCorrelationFitAverage() :
                new LocalCorrelationFitAverage(startingCoordinates.length, options);

        final InferFromMatrix inf = new InferFromMatrix(correlationFit);

        double[] transform;
        try {
            final Visitor visitor = new LazyVisitor(); // always use do-nothing visitor
            transform = inf.estimateZCoordinates(matrix, startingCoordinates, visitor, options);
        } catch (final Exception e) {
            throw new RuntimeException("failed to estimate z coordinates", e);
        }

        return transform;
    }

    /**
     * @param  slicePaths       z-ordered file system paths of slice images to process.
     * @param  comparisonRange  number of adjacent neighbor slices to compare with each slice.
     *
     * @return matrix of pixels representing cross correlation similarity between each slice and its neighbors.
     *
     * @throws IllegalArgumentException
     *   if there are too few or too many slices to process.
     */
    public static RandomAccessibleInterval<DoubleType> buildNCCMatrix(final List<String> slicePaths,
                                                                      final int comparisonRange)
            throws IllegalArgumentException {

        final int sliceCount = slicePaths.size();
        if (sliceCount < 2) {
            throw new IllegalArgumentException("must have at least two slices to evaluate");
        }

        // TODO: replace square matrix image with sliceCount x comparisonRange array (since we are not visualizing)
        final long matrixPixels = (long) sliceCount * sliceCount;
        if (matrixPixels > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("too many slices (" + sliceCount + ") for single matrix");
        }

        // TODO: pass through imagePaths list in chunks so that we don't need to load all pixels up front
        final List<FloatProcessor> imageProcessorList = slicePaths.stream()
                .map(p -> new ImagePlus(p).getProcessor().convertToFloatProcessor())
                .collect(Collectors.toList());

        final FloatProcessor matrixFp = createEmptyMatrix(sliceCount);

        for (int fromSliceIndex = 0; fromSliceIndex < sliceCount; ++fromSliceIndex) {

            final float[] pixelsA = (float[]) imageProcessorList.get(fromSliceIndex).getPixels();

            for (int toSliceIndex = fromSliceIndex + 1;
                 toSliceIndex - fromSliceIndex <= comparisonRange && toSliceIndex < sliceCount;
                 ++toSliceIndex) {

                final float[] pixelsB = (float[]) imageProcessorList.get(toSliceIndex).getPixels();
                final float val = new RealSumFloatNCC(pixelsA, pixelsB).call().floatValue();
                matrixFp.setf(fromSliceIndex, toSliceIndex, val);
                matrixFp.setf(toSliceIndex, fromSliceIndex, val);
            }
        }

        // note: removed stripToMatrix logic for non-square matrices since currently, matrix is always square

        return wrapDouble(new ImagePlus("", matrixFp));
    }

    public static File[] listFilesWithExtension(final File inDirectory,
                                                final String withExtension) {
        return Objects.requireNonNull(inDirectory.listFiles((d, name) -> name.endsWith(withExtension)));
    }

    public static List<String> toSortedAbsolutePaths(final File[] files) {
        return Arrays.stream(files)
                .map(File::getAbsolutePath)
                .sorted()
                .collect(Collectors.toList());
    }

    /**
     * For option descriptions, go to
     * <a href="https://academic.oup.com/bioinformatics/article/33/9/1379/2736362">
     *   this paper
     * </a>, click the dynamically generated "Supplementary data" link, and then scroll to page 7.
     *
     * @return default options for our FIBSEM cases.
     */
    public static Options generateDefaultOptions() {
        final Options options = Options.generateDefaultOptions();

        // values from 2016 paper supplement that match code defaults but are listed here for completeness:
        options.comparisonRange = 10;
        options.nIterations = 100;
        options.shiftProportion = 0.6;
        options.scalingFactorEstimationIterations = 10;
        options.scalingFactorRegularizerWeight = 0.1;
        options.minimumSectionThickness = 0.01;
        options.coordinateUpdateRegularizerWeight = 0.0;
        options.regularizationType = InferFromMatrix.RegularizationType.BORDER;
        options.forceMonotonicity = false;
        options.estimateWindowRadius = -1;       // note: always overridden by code

        // changes:
        options.withReorder = false;             // milled data should always be correctly ordered

        return options;
    }

    /**
     * Main program for running tests ...
     *
     * @param  args  if no arguments are specified, use hard-coded slice directory path;
     *               if one argument is specified, use that as the slice directory path;
     *               if more than one argument is specified, treat all arguments as explicit slice paths.
     */
    public static void main(String[] args) {

        // TODO: manage specification of options and paths better than this hacky test setup ...

        final String imageExtension = ".png";
        final Options baseOptions = generateDefaultOptions();
        final int nLocalEstimates = 1;

        final File[] sliceFiles;
        if (args.length == 0) {
            // test paths: /Users/trautmane/Desktop/zcorr/matt
            //             /Users/trautmane/Desktop/zcorr/matt_Sec08
            //             /Users/trautmane/Desktop/zcorr/crop/05/002
            final File dir = new File("/Users/trautmane/Desktop/zcorr/matt");
            sliceFiles = listFilesWithExtension(dir, imageExtension);
        } else if (args.length == 1) {
            final File dir = new File(args[0]);
            sliceFiles = listFilesWithExtension(dir, imageExtension);
        } else {
            sliceFiles = Arrays.stream(args).map(File::new).toArray(File[]::new);
        }

        final List<String> imagePaths = toSortedAbsolutePaths(sliceFiles);

        final String firstSlicePath = imagePaths.size() > 0 ? imagePaths.get(0) : "";
        final String lastSlicePath = imagePaths.size() > 1 ? imagePaths.get(imagePaths.size() - 1) : "";

        // TODO: output results to file instead of stdout

        System.out.println("\nestimating Z coordinates for " + imagePaths.size() + " slices\n");
        System.out.println("first slice path: " + firstSlicePath);
        System.out.println("last slice path:  " + lastSlicePath);
        System.out.println("\nbase options:" + baseOptions);

        if (imagePaths.size() > 100) {
            System.out.println("WARNING: processing is single threaded and not chunked so this might take a while\n");
        }

        System.out.println("results:\n");

        final double[] transforms = HeadlessZPositionCorrection.estimateZCoordinates(imagePaths,
                                                                                     baseOptions,
                                                                                     nLocalEstimates);
        for (int i = 0; i < transforms.length; i++) {
            System.out.println(i + "," + transforms[i]);
        }
    }
}
