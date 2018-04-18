package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.glu.Project;

import edu.usc.thevillagers.serversideagent.recording.ChangeSet;
import edu.usc.thevillagers.serversideagent.recording.ReplayWorldAccess;
import edu.usc.thevillagers.serversideagent.recording.Snapshot;
import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.renderer.BufferBuilder;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.OpenGlHelper;
import net.minecraft.client.renderer.RenderHelper;
import net.minecraft.client.renderer.Tessellator;
import net.minecraft.client.renderer.entity.RenderManager;
import net.minecraft.client.renderer.texture.TextureMap;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

public class GuiReplay extends GuiScreen {
	
	private WorldRecord record;
	
	private ChangeSet changeSet;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0), prevCamPos = camPos;
	private float camYaw = 0, prevCamYaw = camYaw;
	private float camPitch = -90, prevCamPitch = camPitch;
	
	
	public GuiReplay(File recordFolder) {
		record = new WorldRecord(recordFolder);
	}

	@Override
	public void initGui() {
		super.initGui();
		try {
			record.readInfo();
			Snapshot snapshot = new Snapshot(new File(record.saveFolder, "0.snapshot"));
			snapshot.read();
			snapshot.applyDataToWorld(record);
			changeSet = new ChangeSet(new File(record.saveFolder, "0.changeset"));
			changeSet.read();
			
			camPos = new Vec3d(record.from.add(record.to)).scale(.5);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws IOException {
		super.mouseClicked(mouseX, mouseY, mouseButton);
		Mouse.setGrabbed(mouseButton == 0);
	}
	
	@Override
	public void onGuiClosed() {
		super.onGuiClosed();
		Mouse.setGrabbed(false);
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
		
		if(Mouse.isGrabbed()) {
			camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
			
			camYaw += Mouse.getDX() * .2;
			camPitch += Mouse.getDY() * .2;
		}
		
		if(!changeSet.data.isEmpty())
			for(RecordEvent e : changeSet.data.remove(0)) {
				e.replay(record);
			}
		record.endReplayTick();
		
		mc.getTextureMapBlocks().tick();
	}
	
	private void setupCamera(float partialTicks) {
		GlStateManager.matrixMode(GL11.GL_PROJECTION);
        GlStateManager.loadIdentity();
        Project.gluPerspective(90, (float)this.mc.displayWidth / (float)this.mc.displayHeight, 0.05F, 1000 * 2.0F);
        
        GlStateManager.matrixMode(GL11.GL_MODELVIEW);
		
		
		GlStateManager.loadIdentity();
		GlStateManager.color(1.0F, 1.0F, 1.0F, 1.0F);
        GlStateManager.translate(0, 0.0F, 0F);
        
        float curCamPitch = prevCamPitch + (camPitch - prevCamPitch) * partialTicks;
		float curCamYaw   = prevCamYaw	 + (camYaw   - prevCamYaw  ) * partialTicks;
		GlStateManager.rotate(-curCamPitch, 1, 0, 0);
		GlStateManager.rotate(curCamYaw, 0, 1, 0);
		Vec3d curCamPos = prevCamPos.add(camPos.subtract(prevCamPos).scale(partialTicks));
		GlStateManager.translate(-curCamPos.x, -curCamPos.y, -curCamPos.z);
	}
	
	private void renderBlocks(ReplayWorldAccess world) {
		GlStateManager.enableDepth();
        this.mc.entityRenderer.enableLightmap();
        GlStateManager.enableAlpha();
        GlStateManager.enableBlend();
        GlStateManager.shadeModel(GL11.GL_SMOOTH);
        	
        mc.renderEngine.bindTexture(TextureMap.LOCATION_BLOCKS_TEXTURE);
		
        BufferBuilder buffer = Tessellator.getInstance().getBuffer();
		buffer.begin(7, DefaultVertexFormats.BLOCK);
		buffer.setTranslation(0, 0, 0);
		
		for(BlockPos p : BlockPos.getAllInBoxMutable(record.from, record.to)) {
			IBlockState state = world.getBlockState(p);
			mc.getBlockRendererDispatcher().renderBlock(state, p, world, buffer);
		}
		
		Tessellator.getInstance().draw();
	}
	
	private void renderEntities(ReplayWorldAccess world) {
		RenderHelper.enableStandardItemLighting();
		RenderManager renderManager = mc.getRenderManager();
		renderManager.setPlayerViewY(180);
		renderManager.setRenderShadow(false);
		renderManager.renderViewEntity = world.fakePlayer;
		renderManager.setRenderPosition(0, 0, 0);
		for(Entity e : world.getEntities()) {
			this.mc.entityRenderer.disableLightmap();
			renderManager.renderEntityStatic(e, 1, false);
		}
		renderManager.setRenderShadow(true);
		
		RenderHelper.disableStandardItemLighting();
        GlStateManager.disableRescaleNormal();
        GlStateManager.setActiveTexture(OpenGlHelper.lightmapTexUnit);
        GlStateManager.disableTexture2D();
        GlStateManager.setActiveTexture(OpenGlHelper.defaultTexUnit);
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		drawDefaultBackground();
		
		ReplayWorldAccess world = record.getReplayWorld();
		mc.world = world.fakeWorld;
		mc.player = world.fakePlayer;
		mc.playerController = world.fakePlayerController;
		
		try {
			GlStateManager.pushMatrix();
			setupCamera(partialTicks);
	        renderBlocks(world);
	        renderEntities(world);
			GlStateManager.popMatrix();
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		mc.world = null;
		mc.player = null;
		mc.playerController = null;
		super.drawScreen(mouseX, mouseY, partialTicks);
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}