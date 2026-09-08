package dev.hoonex.adofai.gamepatcher;

import org.jf.dexlib2.DexFileFactory;
import org.jf.dexlib2.Opcode;
import org.jf.dexlib2.Opcodes;
import org.jf.dexlib2.builder.MutableMethodImplementation;
import org.jf.dexlib2.builder.instruction.BuilderInstruction35c;
import org.jf.dexlib2.iface.ClassDef;
import org.jf.dexlib2.iface.DexFile;
import org.jf.dexlib2.iface.Method;
import org.jf.dexlib2.iface.MethodImplementation;
import org.jf.dexlib2.iface.instruction.Instruction;
import org.jf.dexlib2.iface.instruction.ReferenceInstruction;
import org.jf.dexlib2.iface.reference.MethodReference;
import org.jf.dexlib2.immutable.ImmutableClassDef;
import org.jf.dexlib2.immutable.ImmutableDexFile;
import org.jf.dexlib2.immutable.ImmutableMethod;
import org.jf.dexlib2.immutable.reference.ImmutableMethodReference;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Injects one cross-dex call to V240Bootstrap.init() into UnityPlayerActivity.onCreate. */
final class V240DexBootstrapPatcher {
    static final String ACTIVITY = "Lcom/unity3d/player/UnityPlayerActivity;";
    static final String BOOTSTRAP = "Lcom/unity3d/player/V240Bootstrap;";
    private static final String BOOTSTRAP_METHOD = "init";

    static final class Result {
        final boolean alreadyPatched;
        final int classCount;

        Result(boolean alreadyPatched, int classCount) {
            this.alreadyPatched = alreadyPatched;
            this.classCount = classCount;
        }
    }

    static Result patch(File mainDex, File outputDex) throws Exception {
        Opcodes opcodes = Opcodes.forApi(35);
        DexFile main = DexFileFactory.loadDexFile(mainDex, opcodes);
        ClassDef activity = findClass(main, ACTIVITY);
        if (activity == null) throw new IllegalStateException("UnityPlayerActivity class not found");

        Method onCreate = null;
        for (Method method : activity.getMethods()) {
            if (isOnCreate(method)) {
                if (onCreate != null) throw new IllegalStateException("multiple UnityPlayerActivity.onCreate matches");
                onCreate = method;
            }
        }
        if (onCreate == null || onCreate.getImplementation() == null) {
            throw new IllegalStateException("UnityPlayerActivity.onCreate(Bundle) implementation not found");
        }

        boolean alreadyPatched = containsBootstrapInvoke(onCreate.getImplementation());
        Method patchedOnCreate = alreadyPatched ? onCreate : injectBootstrap(onCreate);

        List<Method> activityMethods = new ArrayList<Method>();
        for (Method method : activity.getMethods()) {
            activityMethods.add(method == onCreate ? patchedOnCreate : method);
        }
        ImmutableClassDef patchedActivity = new ImmutableClassDef(
            activity.getType(), activity.getAccessFlags(), activity.getSuperclass(),
            activity.getInterfaces(), activity.getSourceFile(), activity.getAnnotations(),
            activity.getFields(), activityMethods
        );

        List<ClassDef> classes = new ArrayList<ClassDef>();
        for (ClassDef classDef : main.getClasses()) {
            classes.add(ACTIVITY.equals(classDef.getType()) ? patchedActivity : classDef);
        }

        DexFileFactory.writeDexFile(outputDex.getAbsolutePath(), new ImmutableDexFile(main.getOpcodes(), classes));
        if (!outputDex.isFile() || outputDex.length() == 0L) {
            throw new IllegalStateException("patched classes.dex was not written");
        }
        return new Result(alreadyPatched, classes.size());
    }

    static boolean containsBootstrapInvoke(File dexFile) throws Exception {
        DexFile dex = DexFileFactory.loadDexFile(dexFile, Opcodes.forApi(35));
        ClassDef activity = findClass(dex, ACTIVITY);
        if (activity == null) return false;
        for (Method method : activity.getMethods()) {
            if (isOnCreate(method) && method.getImplementation() != null) {
                return containsBootstrapInvoke(method.getImplementation());
            }
        }
        return false;
    }

    private static Method injectBootstrap(Method method) {
        MutableMethodImplementation implementation = new MutableMethodImplementation(method.getImplementation());
        ImmutableMethodReference reference = new ImmutableMethodReference(
            BOOTSTRAP, BOOTSTRAP_METHOD, Collections.<String>emptyList(), "V"
        );
        implementation.addInstruction(0, new BuilderInstruction35c(
            Opcode.INVOKE_STATIC, 0, 0, 0, 0, 0, 0, reference
        ));
        return new ImmutableMethod(
            method.getDefiningClass(), method.getName(), method.getParameters(), method.getReturnType(),
            method.getAccessFlags(), method.getAnnotations(), method.getHiddenApiRestrictions(), implementation
        );
    }

    private static boolean isOnCreate(Method method) {
        if (!ACTIVITY.equals(method.getDefiningClass())) return false;
        if (!"onCreate".equals(method.getName()) || !"V".equals(method.getReturnType())) return false;
        List<? extends CharSequence> parameters = method.getParameterTypes();
        return parameters.size() == 1 && "Landroid/os/Bundle;".contentEquals(parameters.get(0));
    }

    private static boolean containsBootstrapInvoke(MethodImplementation implementation) {
        for (Instruction instruction : implementation.getInstructions()) {
            if (!(instruction instanceof ReferenceInstruction)) continue;
            Object reference = ((ReferenceInstruction) instruction).getReference();
            if (!(reference instanceof MethodReference)) continue;
            MethodReference method = (MethodReference) reference;
            if (BOOTSTRAP.equals(method.getDefiningClass()) &&
                BOOTSTRAP_METHOD.equals(method.getName()) &&
                method.getParameterTypes().isEmpty() &&
                "V".equals(method.getReturnType())) {
                return true;
            }
        }
        return false;
    }

    private static ClassDef findClass(DexFile dex, String type) {
        for (ClassDef classDef : dex.getClasses()) {
            if (type.equals(classDef.getType())) return classDef;
        }
        return null;
    }

    private V240DexBootstrapPatcher() {}
}
