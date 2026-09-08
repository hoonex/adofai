package dev.hoonex.adofai.gamepatcher;

import org.jf.dexlib2.DexFileFactory;
import org.jf.dexlib2.Opcodes;
import org.jf.dexlib2.iface.ClassDef;
import org.jf.dexlib2.iface.DexFile;
import org.jf.dexlib2.immutable.ImmutableClassDef;
import org.jf.dexlib2.immutable.ImmutableDexFile;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Replaces the legacy 2.4 Custom Java file-picker classes inside their existing dex. */
final class DexOverlayPatcher {
    static final String FILE_SELECTOR = "Lcom/unity3d/player/FileSelector;";
    static final String CUSTOM_FILE_CHOOSER = "Lcom/unity3d/player/CustomFileChooser;";
    private static final String FILE_SELECTOR_INNER = "Lcom/unity3d/player/FileSelector$";
    private static final String CUSTOM_FILE_CHOOSER_INNER = "Lcom/unity3d/player/CustomFileChooser$";

    static final class Result {
        final int replacedClasses;
        final int outputClassCount;

        Result(int replacedClasses, int outputClassCount) {
            this.replacedClasses = replacedClasses;
            this.outputClassCount = outputClassCount;
        }
    }

    static Result patch(File sourceDex, File payloadDex, File outputDex) throws Exception {
        Opcodes opcodes = Opcodes.forApi(35);
        DexFile source = DexFileFactory.loadDexFile(sourceDex, opcodes);
        DexFile payload = DexFileFactory.loadDexFile(payloadDex, opcodes);

        Map<String, ClassDef> replacements = new LinkedHashMap<String, ClassDef>();
        for (ClassDef cls : payload.getClasses()) {
            if (isPickerClass(cls.getType())) {
                replacements.put(cls.getType(), cls);
            }
        }
        if (!replacements.containsKey(FILE_SELECTOR) || !replacements.containsKey(CUSTOM_FILE_CHOOSER)) {
            throw new IllegalStateException("2.4 bugfix payload is missing file-picker classes");
        }

        boolean sourceHasSelector = false;
        List<ClassDef> output = new ArrayList<ClassDef>();
        for (ClassDef cls : source.getClasses()) {
            String type = cls.getType();
            if (FILE_SELECTOR.equals(type)) sourceHasSelector = true;
            if (isPickerClass(type)) continue;
            output.add(cls);
        }
        if (!sourceHasSelector) {
            throw new IllegalStateException("legacy FileSelector not found in selected source dex");
        }

        for (ClassDef replacement : replacements.values()) {
            output.add(ImmutableClassDef.of(replacement));
        }

        DexFileFactory.writeDexFile(
            outputDex.getAbsolutePath(),
            new ImmutableDexFile(source.getOpcodes(), output)
        );
        if (!outputDex.isFile() || outputDex.length() == 0L) {
            throw new IllegalStateException("patched dex was not written");
        }

        DexFile verify = DexFileFactory.loadDexFile(outputDex, opcodes);
        boolean selector = false;
        boolean chooser = false;
        int count = 0;
        int pickerClassCount = 0;
        for (ClassDef cls : verify.getClasses()) {
            count++;
            String type = cls.getType();
            if (FILE_SELECTOR.equals(type)) selector = true;
            if (CUSTOM_FILE_CHOOSER.equals(type)) chooser = true;
            if (isPickerClass(type)) pickerClassCount++;
        }
        if (!selector || !chooser || pickerClassCount != replacements.size()) {
            throw new IllegalStateException(
                "patched dex verification failed: expected picker classes=" + replacements.size()
                    + ", actual=" + pickerClassCount
            );
        }
        return new Result(replacements.size(), count);
    }

    private static boolean isPickerClass(String type) {
        return FILE_SELECTOR.equals(type)
            || CUSTOM_FILE_CHOOSER.equals(type)
            || (type.startsWith(FILE_SELECTOR_INNER) && type.endsWith(";"))
            || (type.startsWith(CUSTOM_FILE_CHOOSER_INNER) && type.endsWith(";"));
    }

    private DexOverlayPatcher() {}
}
