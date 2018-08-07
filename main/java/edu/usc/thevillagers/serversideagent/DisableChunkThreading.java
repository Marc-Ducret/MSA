package edu.usc.thevillagers.serversideagent;

import java.util.Map;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodNode;

import net.minecraft.launchwrapper.IClassTransformer;
import net.minecraftforge.fml.relauncher.IFMLLoadingPlugin;

/**
 * Transforms needImediateUpdate method from net.minecraft.client.renderer.chunk.RenderChunk to return true.
 * This effectively disables multi-threading of chunks' mesh computations. It enables the usage of Minecraft's
 * renderer for replaying records.
 */
public class DisableChunkThreading implements IClassTransformer, IFMLLoadingPlugin {

	@Override
	public byte[] transform(String name, String transformedName, byte[] basicClass) {
		try {
			if(transformedName.equals("net.minecraft.client.renderer.chunk.RenderChunk")) {
				System.out.println("Injecting RenderChunk code");
				ClassNode cNode = new ClassNode();
				ClassReader cReader = new ClassReader(basicClass);
				cReader.accept(cNode, 0);
				MethodNode needUpdate = null;
				MethodNode needImediateUpdate = null;
				for(MethodNode method : cNode.methods) {
					switch(method.name) {
					case "needsUpdate":
						needUpdate = method;
						break;
					case "needsImmediateUpdate":
						needImediateUpdate = method;
						break;
					}
				}
				if(needUpdate == null || needImediateUpdate == null) throw new Exception("Cannot find method");
				needImediateUpdate.instructions = new InsnList();
				needImediateUpdate.instructions.add(new InsnNode(Opcodes.ICONST_1));
				needImediateUpdate.instructions.add(new InsnNode(Opcodes.IRETURN));
				ClassWriter cWriter = new ClassWriter(0);
				cNode.accept(cWriter);
				return cWriter.toByteArray();
			}
		} catch(Exception e) {
			System.err.println("Injection failed");
			e.printStackTrace();
		}
		return basicClass;
	}

	@Override
	public String[] getASMTransformerClass() {
		return new String[] {"edu.usc.thevillagers.serversideagent.DisableChunkThreading"};
	}

	@Override
	public String getModContainerClass() {
		return null;
	}

	@Override
	public String getSetupClass() {
		return null;
	}

	@Override
	public void injectData(Map<String, Object> data) {
	}

	@Override
	public String getAccessTransformerClass() {
		return null;
	}
}
