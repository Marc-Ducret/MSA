package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;

import edu.usc.thevillagers.serversideagent.recording.ChangeSet;
import edu.usc.thevillagers.serversideagent.recording.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.ReplayWorldAccess;
import edu.usc.thevillagers.serversideagent.recording.Snapshot;
import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.renderer.BufferBuilder;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.RenderHelper;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

public class GuiReplay extends GuiScreen {
	
	private WorldRecord record;
	
	private ChangeSet changeSet;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0), prevCamPos = camPos;
	private float camYaw = 0, prevCamYaw = camYaw;
	private float camPitch = -90, prevCamPitch = camPitch;

	@Override
	public void initGui() {
		super.initGui();
		System.out.println("New gui replay");
		try {
			record = new WorldRecord(new File("tmp/rec"));
			record.readInfo();
			Snapshot snapshot = new Snapshot(new File(record.saveFolder, "0.snapshot"));
			snapshot.read();
			snapshot.applyDataToWorld((ReplayWorldAccess) record.world, record.from, record.to);
			changeSet = new ChangeSet(new File(record.saveFolder, "0.changeset"));
			changeSet.read();
			
			camPos = new Vec3d(record.from.add(record.to)).scale(.5);
			
			Mouse.setGrabbed(true);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		Vec3d move = Vec3d.ZERO;
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
		if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
		move = move.normalize();
		prevCamPos = camPos;
		prevCamYaw = camYaw;
		prevCamPitch = camPitch;
		
		camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
		
		camYaw += Mouse.getDX() * .2;
		camPitch += Mouse.getDY() * .2;
		
		if(!changeSet.data.isEmpty())
			for(RecordEvent e : changeSet.data.remove(0)) {
				e.replay(record);
			}
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawDefaultBackground();
		
		try {
			GlStateManager.matrixMode(GL11.GL_PROJECTION);
            GlStateManager.loadIdentity();
            Project.gluPerspective(90, (float)this.mc.displayWidth / (float)this.mc.displayHeight, 0.05F, 1000 * 2.0F);
            
            GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			GlStateManager.color(1.0F, 1.0F, 1.0F, 1.0F);
	        GlStateManager.translate(0, 0.0F, 0F);
	        
            GlStateManager.enableDepth();
            RenderHelper.disableStandardItemLighting();
            this.mc.entityRenderer.enableLightmap(); 
            GlStateManager.shadeModel(7425);
            GlStateManager.enableAlpha();
            GlStateManager.enableBlend();
            mc.renderEngine.bindTexture(TextureMap.LOCATION_BLOCKS_TEXTURE);
			
			BufferBuilder buffer = Tessellator.getInstance().getBuffer();
			buffer.begin(7, DefaultVertexFormats.BLOCK);
			buffer.setTranslation(0, 0, 0);
			buffer.noColor();
			
			float curCamPitch = prevCamPitch + (camPitch - prevCamPitch) * partialTicks;
			float curCamYaw   = prevCamYaw	 + (camYaw   - prevCamYaw  ) * partialTicks;
			GlStateManager.rotate(-curCamPitch, 1, 0, 0);
			GlStateManager.rotate(curCamYaw, 0, 1, 0);
			Vec3d curCamPos = prevCamPos.add(camPos.subtract(prevCamPos).scale(partialTicks));
			GlStateManager.translate(-curCamPos.x, -curCamPos.y, -curCamPos.z);
			
			for(BlockPos p : BlockPos.getAllInBoxMutable(record.from, record.to)) {
				mc.getBlockRendererDispatcher().renderBlock(record.world.getBlockState(p), p, record.world, Tessellator.getInstance().getBuffer());
			}
			Tessellator.getInstance().draw();
			
			GlStateManager.popMatrix();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}