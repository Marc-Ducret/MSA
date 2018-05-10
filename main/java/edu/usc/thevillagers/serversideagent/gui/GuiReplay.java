package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.PriorityBlockingQueue;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayerClient;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.RenderGlobal;
import net.minecraft.client.renderer.chunk.ChunkCompileTaskGenerator;
import net.minecraft.client.renderer.chunk.ChunkRenderDispatcher;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLivingBase;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.init.Items;
import net.minecraft.item.ItemStack;
import net.minecraft.util.math.Vec3d;
import net.minecraftforge.fml.client.config.GuiSlider;

/**
 * Display for {@link WorldRecordReplayerClient}
 */
public class GuiReplay extends GuiScreen {
	
	private WorldRecordReplayerClient record;
	private int speed;
	
	private Vec3d camPos = Vec3d.ZERO.addVector(0, 10, 0);
	private float camYaw = 0;
	private float camPitch = -90;
	
	private Entity followedEntity;
	
	private GuiSlider seekSlider;
	
	public GuiReplay(File recordFolder) {
		record = new WorldRecordReplayerClient(recordFolder);
		speed = 1;
	}

	@Override
	public void initGui() {
		super.initGui();
		try {
			if(record.world == null) {
				record.readInfo();
				
				record.seek(0);
				
				camPos = new Vec3d(record.from.add(record.to)).scale(.5);
				
				mc.renderGlobal.setWorldAndLoadRenderers((WorldClient) record.world);
				ChunkRenderDispatcher dispatcher =
						ServerSideAgentMod.<ChunkRenderDispatcher>getPrivateField(RenderGlobal.class, "renderDispatcher", mc.renderGlobal);
				ServerSideAgentMod.<PriorityBlockingQueue<ChunkCompileTaskGenerator>>setPrivateField(
						ChunkRenderDispatcher.class, "queueChunkUpdates", dispatcher,
						new PriorityBlockingQueue<ChunkCompileTaskGenerator>() {
							private static final long serialVersionUID = 1L;
							
							@Override
							public boolean offer(ChunkCompileTaskGenerator e) {
								return false;
							}
						});
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		addButton(new GuiButton(0, 0, height-20, 20, 20, "<"));
		addButton(new GuiButton(1, 50, height-20, 20, 20, ">"));
		addButton(new GuiButton(2, 75, height-20, 20, 20, "+1"));
		seekSlider = new GuiSlider(3, width/2 - 100, height-20, 200, 20, "", " s", 0, record.duration / 20F, 10, true, true) {
			@Override
			public void mouseReleased(int x, int y) {
				super.mouseReleased(x, y);
				mc.addScheduledTask(() -> {
					try {
						followedEntity = null;
						record.seek((int) Math.round(getValue() * 20));
						mc.renderGlobal.setWorldAndLoadRenderers((WorldClient) record.world);
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			}
		};
		addButton(seekSlider);
	}
	
	@Override
	protected void actionPerformed(GuiButton button) throws IOException {
		super.actionPerformed(button);
		switch(button.id) {
		case 0:
			if(speed > 1)
				speed /= 2;
			else
				speed = 0;
			break;
		case 1:
			if(speed < 2048)
				speed *= 2;
			if(speed == 0)
				speed = 1;
			break;
		case 2:
			try {
				record.endReplayTick();
			} catch (Exception e) {
				throw new RuntimeException("Replay tick failure", e);
			}
			break;
		}
	}
	
	@Override
	protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws IOException {
		super.mouseClicked(mouseX, mouseY, mouseButton);
		for(GuiButton b : buttonList)
			if(b.isMouseOver())
				return;
		Mouse.setGrabbed(mouseButton == 0);
	}
	
	@Override
	protected void keyTyped(char typedChar, int keyCode) throws IOException {
		super.keyTyped(typedChar, keyCode);
		if(keyCode == mc.gameSettings.keyBindInventory.getKeyCode()) {
			if(followedEntity == null) {
				Vec3d rayStart = record.player.getPositionVector();
				Vec3d rayEnd = rayStart.add(record.player.getLookVec().scale(5));
				for(Entity e : record.world.getLoadedEntityList()) { //TODO use world ray trace ;)
					if(e.getEntityBoundingBox() != null && e.getEntityBoundingBox().calculateIntercept(rayStart, rayEnd) != null)
						followedEntity = e;
				}
			} else {
				followedEntity = null;
			}
		}
	}
	
	@Override
	public void onGuiClosed() {
		super.onGuiClosed();
		Mouse.setGrabbed(false);
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		if(followedEntity != null) {
			camPos = followedEntity.getPositionVector().addVector(0, followedEntity.getEyeHeight(), 0);
			camPitch = -followedEntity.rotationPitch;
			camYaw = followedEntity.rotationYaw + 180;
		} else {
			Vec3d move = Vec3d.ZERO;
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
			if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
			move = move.normalize();
			
			if(Mouse.isGrabbed()) {
				camPos = camPos.add(move.rotateYaw((float)Math.toRadians(-camYaw)).scale(.5));
				
				camYaw += Mouse.getDX() * .2;
				camPitch += Mouse.getDY() * .2;
			}
		}
		
		EntityPlayerSP player = record.player;
		player.prevRotationPitch 	= player.rotationPitch;
		player.prevRotationYaw 		= player.rotationYaw;
		player.lastTickPosX			= player.posX;
		player.lastTickPosY 		= player.posY;
		player.lastTickPosZ 		= player.posZ;
		player.rotationYaw 			= 180+camYaw;
		player.rotationPitch 		= -camPitch;
		player.posX					= camPos.x;
		player.posY 				= camPos.y;
		player.posZ					= camPos.z;
		player.turn(0, 0); //TODO useful?
		
		try {
			for(int i = 0; i < speed; i ++) {
				record.endReplayTick();
				for(Entity e : record.world.getLoadedEntityList()) {
					if(e instanceof EntityPlayer) {
						EntityPlayer p = (EntityPlayer) e;
						try {
							Method method = EntityLivingBase.class.getDeclaredMethod("updateArmSwingProgress");
							method.setAccessible(true);
							method.invoke(p);
						} catch(Exception ex) {
							ex.printStackTrace();
						}
					}
				}
			}
			seekSlider.setValue(record.currentTick / 20F);
			seekSlider.updateSlider();
		} catch (InterruptedException | ExecutionException e) {
			throw new RuntimeException("Replay tick failure", e);
		}
		
		mc.getTextureMapBlocks().tick();
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		drawDefaultBackground();
		
		mc.world = (WorldClient) record.world;
		mc.player = record.player;
		mc.playerController = record.playerController;
		mc.setRenderViewEntity(mc.player);
		
		try {
			GlStateManager.matrixMode(GL11.GL_PROJECTION);
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			mc.entityRenderer.renderWorld(1, System.nanoTime() + 100000); //TODO interpolation?
//			setupCamera(partialTicks);
//	        renderBlocks(world);
//	        renderEntities(world);
	        GlStateManager.matrixMode(GL11.GL_PROJECTION);
			GlStateManager.popMatrix();
			GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			GlStateManager.popMatrix();
			GlStateManager.disableDepth();
		} catch(Exception e) {
			e.printStackTrace();
		}
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawCenteredString(mc.fontRenderer, speed+"*", 35, height-14, 0xFFFFFF);
		if(followedEntity != null) {
			drawCenteredString(mc.fontRenderer, followedEntity.getName(), width / 2, 2, 0xFFFFFF);
		}
        itemRender.renderItemAndEffectIntoGUI(new ItemStack(Items.CLOCK), width-18, height-18);
		mc.world = null;
		mc.player = null;
		mc.playerController = null;
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}